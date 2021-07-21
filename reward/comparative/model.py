import gin
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer.transformer import \
    Unitransformer, get_vocab_embedding_cls, make_layer_stack

from reward.comparative.loss_fn import comparative_reward_pair_loss

class ScalarOutputUnitransformer(Unitransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_vocab_size = None
        self.autoregressive = False
        self._reward_dim =  mtf.Dimension("reward", 1)

    def _call_internal(self, context, inputs, targets=None):
        """
        Overrrides Unitransformer._call_internal
        Adds losses to context!
        """
        mesh = inputs.mesh
        if self.ensemble_dim and self.ensemble_dim not in inputs.shape.dims:
            # Training an ensemble where all models are trained on the same examples.
            inputs = mtf.broadcast(inputs, [self.ensemble_dim] + inputs.shape.dims)
            if targets:
                targets = mtf.broadcast(
                    targets, [self.ensemble_dim] + targets.shape.dims)
        if "embedding" in context.shared_params:
            vocab_embedding = context.shared_params["embedding"]
        else:
            vocab_embedding = get_vocab_embedding_cls()(
                mesh,
                self.input_vocab_dim,
                self.model_dim,
                context.variable_dtype,
                name="embedding",
                ensemble_dim=self.ensemble_dim)
        x = vocab_embedding.ids_to_embedding(inputs)
        if self.positional_embedding:
            if "positional_embedding" in context.shared_params:
                pos_emb_var = context.shared_params["positional_embedding"]
            else:
                pos_emb_var = mtf.layers.embedding_weights(
                    mesh, self.max_length_dim, self.model_dim, context.variable_dtype,
                    "positional_embedding", ensemble_dim=self.ensemble_dim)
            if (context.length_dim is not None and
                context.length_dim.size > self.max_length_dim.size):
                message = (
                    "Length dimenison exceeds size of positional embedding table. "
                    "length_dim.size > max_length_dim.size %s vs %s."
                    % (context.length_dim, self.max_length_dim))
                if context.position_is_default:
                    # Definitely getting overflow in this case.
                    raise ValueError(message)
                else:
                    tf.logging.warning(
                        message +
                        " This may be OK if there are several shorter sequences packed "
                        "together.  Otherwise, the later positions will get zeros.")
            if context.position_is_default:
                pos_emb = mtf.rename_dimension(
                    mtf.slice(pos_emb_var, 0, context.length_dim.size,
                            self.max_length_dim.name),
                    self.max_length_dim.name, context.length_dim.name)
            else:
                pos_emb = mtf.gather(
                    pos_emb_var, context.position, self.max_length_dim,
                    output_shape=x.shape)
            x += pos_emb
        x = self.layer_stack.call(context, x)                                                                                                                                                                                                                                                                                        
        logits = mtf.layers.dense(
            x,
            new_dims=self._reward_dim,
            reduced_dims=[d for d in x.shape.dims if d.name=="d_model"],
            use_bias=False,
            variable_dtype=context.variable_dtype,
            name="reward_head"
        )
        # Squeeze out size 1 reward dimension
        squeezed_shape = mtf.Shape([d for d in logits.shape.dims if d.name != "reward"])
        logits = mtf.reshape(logits, squeezed_shape, name="squeeze_reward_dim")
        """
        if self.shared_embedding_and_softmax_weights:
            logits = vocab_embedding.hidden_to_logits(x)
        else:
            logits = mtf.layers.dense(
                x, self.output_vocab_dim, use_bias=False,
                variable_dtype=context.variable_dtype,
                reduced_dims=x.shape.dims[-1:],
                name="logits")
        """
        if targets is not None and context.losses is not None:
            context.losses.append(

            )
            # context.losses.append(
            #     self._compute_loss(context, logits, targets, self.output_vocab_dim))
        return logits

@gin.configurable
def make_reward_bitransformer(
    input_vocab_size=gin.REQUIRED,
    output_vocab_size=gin.REQUIRED,
    layout=None,
    mesh_shape=None,
    encoder_name="encoder",
    decoder_name="decoder"
    ):
    with gin.config_scope("encoder"):
        encoder = Unitransformer(
            layer_stack=make_layer_stack(),
            input_vocab_size=input_vocab_size,
            output_vocab_size=None,
            autoregressive=False,
            name=encoder_name,
            layout=layout,
            mesh_shape=mesh_shape
        )