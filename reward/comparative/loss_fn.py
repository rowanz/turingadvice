import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf

def comparative_paired_rewards_loss(paired_rewards, ans_pair_dim):
    """
    Loss function for the comparative reward model from Learning to Summarize
    from Human Feedback (Stiennon et al.).

    Parameters
    ----------
    paired_rewards : mtf.Tensor
        Output of the reward model for two responses to the same prompt (e.g.
        two summaries of the same text). The second value along the
        `ans_pair_dim` axis is assumed to correspond to the human preferred
        response. Shape should be [ans_pair_dim, batch_dim].
    ans_pair_dim : mtf.Dimension
        Dimension of the answer pair. ans_pair_dim.size = 2

    Returns
    -------
    loss : mtf.Tensor
        Aggregative mini-batch loss with shape [].
    """
    tf_diff_filter = tf.convert_to_tensor([-1, 1], dtype=paired_rewards.dtype)
    diff_filter = mtf.import_tf_tensor(
        paired_rewards.mesh,
        tf_diff_filter,
        shape=[ans_pair_dim]
    )
    diff = mtf.reduce_sum(paired_rewards * diff_filter, reduced_dim=ans_pair_dim)
    return mtf.reduce_mean(-mtf.log(mtf.sigmoid(diff)))
