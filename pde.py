import torch

class PDE:
    def derivative(self, y_diff: torch.Tensor, x_diff: torch.Tensor, order=1):
        result = y_diff
        for _ in range(order):
            result = torch.autograd.grad(
                outputs= result,
                inputs = x_diff,
                grad_outputs=torch.ones_like(result),
                create_graph= True,
                allow_unused=True
            )[0]
        return result
    
    def residual(self, fn, vars, real_vars):
        return fn(vars, real_vars, self.derivative)
