import torch

def f_fn(u, s):
    """
    Quadratic energy cost: f(u) = ||u||^2
    Args:
        u: [batch_size, action_dim] tensor of joint torques
    Returns:
        [batch_size] scalar cost per sample
    """
    return 0.01*torch.sum(u ** 2, dim=1)

def g_fn(u, s, u_max=0.6, u_min=-0.6):
    # g_upper = u - u_max  # u - umax <= 0
    # g_lower = -u + u_min  # -u + umin <= 0
    g_height = (0.8 - s[:, 1]).unsqueeze(-1)  # height constraint, assuming s[:, 1] is the height
    g_pitch_upper = (s[:, 2] - 0.2).unsqueeze(-1)
    g_pitch_lower = (-s[:, 2] - 0.2).unsqueeze(-1)
    return torch.cat([g_height, g_pitch_upper, g_pitch_lower], dim=1)

def h_fn(u, s):
    return (2*s[:,2] + s[:,3] + s[:,4]).unsqueeze(-1)

