class BestOfNGenerator():
    def __init__(
        self, t5_model, t5_model_ckpt_steps, N, sampling_keep_top_p
        ):
        self.t5_model = t5_model
        self.model_ckpt_steps = t5_model_ckpt_steps
        self.N = N
        self.sampling_keep_top_p = sampling_keep_top_p
    
    def predict_from_instances(instances):
        """
        Args:
        instances: [dict]
            A list of dicts with keys "title", "selftext", "subreddit", "date"
        
        Returns:
        advices: [str]
            List of generations for each instance. Same length as `instances`
        """
        return [f"Advice {i}" for i, _ in enumerate(instances)]