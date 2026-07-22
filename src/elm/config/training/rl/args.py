import argparse

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--rl_algo", type=str, default="sapo", choices=["sapo"], help="RL policy-loss algorithm")
    parser.add_argument("--rl_group_size", type=int, default=4, help="G: rollouts per prompt for group-relative advantage")
    parser.add_argument("--rl_max_new_tokens", type=int, default=512, help="Max new tokens per rollout")
    parser.add_argument("--rl_temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--rl_top_p", type=float, default=1.0, help="Nucleus sampling p")
    parser.add_argument("--rl_tau_pos", type=float, default=1.0, help="SAPO temperature for positive advantages")
    parser.add_argument("--rl_tau_neg", type=float, default=1.05, help="SAPO temperature for negative advantages")
    parser.add_argument("--rl_loss_agg_mode", type=str, default="seq-mean-token-mean",
                        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"])
    return parser.parse_args()