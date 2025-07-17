import torch
import torch.nn.functional as F

def compute_kkt_loss(u, s, f_fn, lambda_, mu_, g_fn=None, h_fn=None ):
    """
    Compute KKT-based loss function.
    Arguments:
        u        : action variable (requires_grad=True), shape [batch_size, action_dim]
        s        : state variable (optional, can be used in f_fn, g_fn, h_fn), shape [batch_size, state_dim]
        f_fn     : function returning cost f(u, s), shape [batch_size]
        g_fn     : function returning inequality constraints g(u, s), shape [batch_size, num_ineq]
        h_fn     : function returning equality constraints h(u, s), shape [batch_size, num_eq]
        lambda_  : Lagrange multipliers for g(u, s), shape [batch_size, num_ineq]
        mu_      : Lagrange multipliers for h(u, s), shape [batch_size, num_eq]
    Returns:
        scalar KKT loss
    """
    # Compute Lagrangian
    f_u = f_fn(u, s)  # should return shape [batch_size]
    g_u = g_fn(u, s) if g_fn is not None else torch.zeros_like(lambda_)
    h_u = h_fn(u, s) if h_fn is not None else torch.zeros_like(mu_)

    # Compute gradients of the Lagrangian: ∇_u [f(u) + λᵀg(u) + μᵀh(u)]
    # L = f_u + (lambda_ * g_u).sum(dim=1) + (mu_ * h_u).sum(dim=1)
    L = f_u 
    grad = torch.autograd.grad(outputs=L, inputs=u, grad_outputs=torch.ones_like(f_u),
                               create_graph=True, retain_graph=True)[0]

    # KKT conditions
    stationarity = torch.sum(grad**2, dim=1)
    primal_ineq = torch.sum(F.relu(g_u)**2, dim=1)
    primal_eq = torch.sum(h_u**2, dim=1)
    slackness = torch.sum((lambda_ * g_u)**2, dim=1)
    dual_feas = torch.sum(F.relu(-lambda_)**2, dim=1)

    kkt_loss = stationarity + primal_ineq + primal_eq + slackness + dual_feas
    return kkt_loss.mean()
