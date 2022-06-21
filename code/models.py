from typing import Union, List, Callable, Optional
import torch
from torch import Tensor
from torch import nn
from torch.optim import _functional as F
from torch.optim import Optimizer

class FeedForward(nn.Module):
    """
    Class for feedforward neural network model. Takes a list of pytorch
    tensors holding the weight initializations and
    ties these together into a trainable neural network.
    """

    def __init__(self, layer_weights: List[Tensor], biases: List[Tensor],
                 nonlinearities: List[Callable],
                 layer_train=None, bias_train=None):
        """

        Parameters
        ----------
        layer_weights : List[Tensor]
            List of the layer initializations.
        biases : List[Tensor]
            List of the bias initializations.
        nonlinearities : List[Callable]
            List of the nonlinearities used in the layers.
        layer_train : List[bool]
            layer_train[k] specifies if layer k is trained or fixed
        bias_train : List[bool]
            layer_train[k] specifies if layer k is trained or fixed
        """
        super().__init__()
        if layer_train is None:
            layer_train = [True]*len(layer_weights)
        if bias_train is None:
            bias_train = [True]*len(biases)
        self.layer_weights = nn.ParameterList(
            [nn.Parameter(layer, requires_grad=train) for layer, train in
             zip(layer_weights, layer_train)])
        self.biases = nn.ParameterList(
            [nn.Parameter(bias, requires_grad=train) for bias, train in
             zip(biases,
                 bias_train)])
        self.nonlinearities = nonlinearities

    def forward(self, inputs: Tensor):
        hid = inputs
        for layer, nonlinearity, bias in zip(self.layer_weights,
                                             self.nonlinearities, self.biases):
            hid = nonlinearity(hid@layer + bias)
        return hid

    def get_pre_activations(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = inputs
            pre_activations = []
            for layer, nonlinearity, bias in zip(self.layer_weights,
                                                 self.nonlinearities,
                                                 self.biases):
                pre_activation = hid@layer + bias
                hid = nonlinearity(pre_activation)
                pre_activations.append(detacher(pre_activation))
            return pre_activations

    def get_post_activations(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = inputs
            post_activations = []
            for layer, nonlinearity, bias in zip(self.layer_weights,
                                                 self.nonlinearities,
                                                 self.biases):
                hid = nonlinearity(hid@layer + bias)
                post_activations.append(detacher(hid))
            return post_activations

    def get_activations(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = inputs
            activations = []
            for layer, nonlinearity, bias in zip(self.layer_weights,
                                                 self.nonlinearities,
                                                 self.biases):
                pre_activation = hid@layer + bias
                hid = nonlinearity(pre_activation)
                activations.append(detacher(pre_activation))
                activations.append(detacher(hid))
            return activations

# noinspection PyArgumentList
class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN). This is a "vanilla" implementation with
    the typical machine-learning style
    equations:

        h_{t+1} = nonlinearity(h_{t} @ recurrent_weights + recurrent_bias)
        --  hidden unit update
    """

    def __init__(self, input_weights: Tensor, recurrent_weights: Tensor,
                 output_weights: Tensor,
                 recurrent_bias: Tensor, output_bias: Tensor,
                 nonlinearity: Optional[Union[str, Callable]],
                 hidden_unit_init: Optional[Union[str, Callable]] = None,
                 train_input: bool = False,
                 train_recurrent: bool = True, train_output: bool = True,
                 train_recurrent_bias: bool = True,
                 train_output_bias: bool = True,
                 output_over_recurrent_time: bool = False,
                 dropout_p=0, unit_injected_noise=0):
        """
        Parameters
        ----------
        input_weights : Tensor
            Input weight initialization.
        recurrent_weights : Tensor
            Recurrent weight initialization.
        output_weights : Tensor
            Output weight initialization.
        recurrent_bias : Tensor
            Recurrent bias vector initialization.
        output_bias : Tensor
            Output bias vector initialization.
        nonlinearity : Optional[Union[str, Callable]]
            The nonlinearity to use for the hidden unit activation function.
        hidden_unit_init : Optional[Union[str, Callable]]
            Initial value for the hidden units. The network is set to this
            value at the beginning of every input
            batch. Todo: make it so the hidden state can carry over input
            batches.
        train_input : bool
            True: train the input weights, i.e. set requires_grad = True for
            the input weights. False: keep the input
            weights fixed to their initial value over training.
        train_recurrent : bool
            True: train the recurrent weights. False: keep the recurrent
            weights fixed to their initial value over training.
        train_output : bool
            True: train the output weights. False: keep the output weights
            fixed to their initial value over
            training.
        train_recurrent_bias : bool
            True: train the recurrent bias. False: keep the recurrent bias
            fixed to its initial value over training.
        train_output_bias : bool
            True: train the output bias. False: keep the output bias fixed to
            its initial value over training.
        output_over_recurrent_time : bool
            True: Return network output over the recurrent timesteps. False:
            Only return the network output at the
            last timestep.
        dropout_p : float
            Probability value for dropout applied to the hidden units of the
            feedforward network or recurrent units at each recurrent timestep.
            Default: 0. If 0, a dropout layer isn't added.
        unit_injected_noise : float
            Magnitude of i.i.d Gaussian noise to inject in each unit of each
            hidden layer or on each recurrent timestep. Default: 0.
        """

        super().__init__()
        if isinstance(nonlinearity, Callable):
            self.nonlinearity = nonlinearity
        elif isinstance(nonlinearity, str):
            if nonlinearity == 'tanh' or nonlinearity == 'Tanh':
                self.nonlinearity = torch.tanh
            elif nonlinearity == 'relu' or nonlinearity == 'ReLU':
                def relu(x):
                    return torch.clamp(x, min=0)

                self.nonlinearity = relu
            else:
                raise AttributeError("nonlinearity not recognized.")
        else:
            raise AttributeError("nonlinearity not recognized.")

        if hidden_unit_init is None:
            self.hidden_unit_init = nn.Parameter(
                torch.zeros(recurrent_weights.shape[0]),
            requires_grad=False)
            # self.hidden_unit_init = torch.zeros(recurrent_weights.shape[0])
        elif isinstance(hidden_unit_init, Tensor):
            self.hidden_unit_init = nn.Parameter(hidden_unit_init.clone(),
                                                requires_grad=False)
        else:
            raise AttributeError("hidden_unit_init option not recognized.")

        if train_input:
            self.Win = nn.Parameter(input_weights.clone(), requires_grad=True)
        else:
            self.Win = nn.Parameter(input_weights.clone(), requires_grad=False)

        if train_recurrent:
            self.Wrec = nn.Parameter(recurrent_weights.clone(),
                                     requires_grad=True)
        else:
            self.Wrec = nn.Parameter(recurrent_weights.clone(),
                                     requires_grad=False)

        if train_output:
            self.Wout = nn.Parameter(output_weights.clone(), requires_grad=True)
        else:
            self.Wout = nn.Parameter(output_weights.clone(),
                                     requires_grad=False)

        if train_recurrent_bias:
            self.brec = nn.Parameter(recurrent_bias.clone(), requires_grad=True)
        else:
            self.brec = nn.Parameter(recurrent_bias.clone(),
                                     requires_grad=False)

        if train_output_bias:
            self.bout = nn.Parameter(output_bias.clone(), requires_grad=True)
        else:
            self.bout = nn.Parameter(output_bias.clone(), requires_grad=False)


        self.output_over_recurrent_time = output_over_recurrent_time

        self.dropout_p = dropout_p
        self.unit_injected_noise = unit_injected_noise
        dropout = torch.nn.Dropout(dropout_p)
        def gauss_noise_inject(x):
            xi = unit_injected_noise * torch.randn(*x.shape)
            return x + xi
        if self.dropout_p > 0 and self.unit_injected_noise > 0:
            def noise_inject(x, training=True):
                if training:
                    return dropout(gauss_noise_inject(x))
                else:
                    return x
        elif self.dropout_p > 0:
            def noise_inject(x, training=True):
                if training:
                    return dropout(x)
                else:
                    return dropout(x)
        elif self.unit_injected_noise > 0:
            def noise_inject(x, training=True):
                if training:
                    return gauss_noise_inject(x)
                else:
                    return x
        else:
            def noise_inject(x, training=True):
                return x
        self.noise_inject = noise_inject

    def forward(self, inputs: Tensor):
        hid = self.hidden_unit_init
        if self.output_over_recurrent_time:
            out = torch.zeros(inputs.shape[0], inputs.shape[1],
                              self.Wout.shape[-1], device=inputs.device)
            for i0 in range(inputs.shape[1]):
                preactivation = (hid@self.Wrec +
                                 inputs[:, i0]@self.Win + self.brec)
                hid = self.noise_inject(self.nonlinearity(preactivation),
                                        self.training)
                out[:, i0] = hid@self.Wout + self.bout
            return out
        else:
            for i0 in range(inputs.shape[1]):
                preactivation = (hid@self.Wrec +
                                 inputs[:, i0]@self.Win + self.brec)
                hid = self.noise_inject(self.nonlinearity(preactivation),
                                        self.training)
            out = hid@self.Wout + self.bout
            return out

    def get_pre_activations(self, inputs: Tensor):
        hid = self.hidden_unit_init
        preactivations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid@self.Wrec + inputs[:, i0]@self.Win + self.brec
            hid = self.nonlinearity(preactivation)
            preactivations.append(preactivation.detach())
        out = hid@self.Wout + self.bout
        preactivations.append(out.detach())
        return preactivations

    def get_post_activations(self, inputs: Tensor):
        hid = self.hidden_unit_init
        postactivations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid@self.Wrec + inputs[:,
                                              i0]@self.Win + self.brec
            hid = self.nonlinearity(preactivation)
            postactivations.append(hid.detach())
        out = hid@self.Wout + self.bout
        postactivations.append(out.detach())
        return postactivations

    def get_activations(self, inputs: Tensor):
        hid = self.hidden_unit_init
        activations = []
        for i0 in range(inputs.shape[1]):
            preactivation = hid@self.Wrec + inputs[:,
                                              i0]@self.Win + self.brec
            hid = self.nonlinearity(preactivation)
            activations.append(preactivation.detach())
            activations.append(hid.detach())
        out = hid@self.Wout + self.bout
        activations.append(out.detach())
        return activations


class SompolinskyRNN(RNN):
    """
    Recurrent Neural Network (RNN) with style dynamics as used in Sompolinsky
    et al. 1988:

    h' = -h + nonlinearity(h)@Wrec + input@Win + recurrent_bias.

    These are discretized via forward Euler method to get the update

    h_{t+1} = h_{t} + dt(-h_{t} + nonlinearity(h_{t}) @ Wrec + input_{t}@Win
    + recurrent_bias)

    Here h is like a current input (membrane potential) and nonlinearity(h_{
    t}) is like a "firing rate".
    """

    def __init__(self, input_weights: Tensor, recurrent_weights: Tensor,
                 output_weights: Tensor,
                 recurrent_bias: Tensor, output_bias: Tensor,
                 nonlinearity: Optional[Union[str, Callable]],
                 hidden_unit_init: Optional[Union[str, Tensor]] = None,
                 train_input: bool = False,
                 train_recurrent: bool = True, train_output: bool = True,
                 train_recurrent_bias: bool = True,
                 train_output_bias: bool = True, dt: float = .01,
                 output_over_recurrent_time: bool = False):
        """
        Parameters
        ----------
        input_weights : Tensor
            Input weight initialization.
        recurrent_weights : Tensor
            Recurrent weight initialization.
        output_weights : Tensor
            Output weight initialization.
        recurrent_bias : Tensor
            Recurrent bias vector initialization.
        output_bias : Tensor
            Output bias vector initialization.
        nonlinearity : Optional[Union[str, Callable]]
            The nonlinearity to use for the hidden unit activation function.
        hidden_unit_init : Optional[Union[str, Callable]]
            Initial value for the hidden units. The network is set to this
            value at the beginning of every input
            batch. Todo: make it so the hidden state can carry over input
            batches.
        train_input : bool
            True: train the input weights, i.e. set requires_grad = True for
            the input weights. False: keep the input
            weights fixed to their initial value over training.
        train_recurrent : bool
            True: train the recurrent weights. False: keep the recurrent
            weights fixed to their initial value over training.
        train_output : bool
            True: train the output weights. False: keep the output weights
            fixed to their initial value over
            training.
        train_recurrent_bias : bool
            True: train the recurrent bias. False: keep the recurrent bias
            fixed to its initial value over training.
        train_output_bias : bool
            True: train the output bias. False: keep the output bias fixed to
            its initial value over training.
        output_over_recurrent_time : bool
            True: Return network output over the recurrent timesteps. False:
            Only return the network output at the
            last timestep.
        """
        super().__init__(input_weights, recurrent_weights, output_weights,
                         recurrent_bias, output_bias, nonlinearity,
                         hidden_unit_init, train_input, train_recurrent,
                         train_output, train_recurrent_bias,
                         train_output_bias, output_over_recurrent_time)
        self.dt = dt

    def forward(self, inputs: Tensor):
        hid = self.hidden_unit_init
        if self.output_over_recurrent_time:
            # out = [hid]
            out = torch.zeros(inputs.shape[0], inputs.shape[1],
                              self.Wout.shape[-1], device=inputs.device)
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(
                            self.nonlinearity(hid)@self.Wrec + inputs[:,
                                                                 i0]@self.Win + self.brec)
                out[:, i0] = hid@self.Wout + self.bout
            return out
        else:
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(
                            self.nonlinearity(hid)@self.Wrec + inputs[:,
                                                                 i0]@self.Win + self.brec)
            out = hid@self.Wout + self.bout
            return out

    def get_currents(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = self.hidden_unit_init
            currents = []
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(
                            self.nonlinearity(hid)@self.Wrec + inputs[:,
                                                                 i0]@self.Win + self.brec)
                currents.append(hid)
            return currents

    def get_firing_rates(self, inputs: Tensor, detach: bool = True):
        if detach:
            def detacher(hid):
                return hid.detach().clone()
        else:
            def detacher(hid):
                return hid.clone()
        with torch.set_grad_enabled(not detach):
            hid = self.hidden_unit_init
            firing_rates = []
            for i0 in range(inputs.shape[1]):
                hid = hid + self.dt*(
                            self.nonlinearity(hid)@self.Wrec + inputs[:,
                                                                 i0]@self.Win + self.brec)
                firing_rates.append(self.nonlinearity(hid))
            return firing_rates

    def get_activations(self, inputs: Tensor, detach: bool = True):
        raise AttributeError(
            "get_activations is not implemented for this model. Try get_currents or get_firing_rates.")

    def get_pre_activations(self, inputs: Tensor,
                            detach: bool = True):  # Alias for compatibility with other models
        return self.get_currents(inputs, detach)

    def get_post_activations(self, inputs: Tensor,
                             detach: bool = True):  # Alias for compatibility with other models
        return self.get_firing_rates(inputs, detach)


class NoisySGD(torch.optim.SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, grad_noise=0,
                 noise_p_idx=None, scale_noise=None):
        super().__init__(params, lr, momentum, dampening,
                         weight_decay, nesterov)
        self.grad_noise = grad_noise
        self.noise_p_idx = noise_p_idx
        self.scale_noise = scale_noise

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if self.grad_noise > 0.0:
                    if self.noise_p_idx is not None:
                        if k in self.noise_p_idx:
                            noise = torch.randn(*p.grad.shape)
                            if self.scale_noise:
                                ps = p.grad.norm()
                                noise = ps*noise/noise.norm()
                            d_p = p.grad + self.grad_noise*noise
                        else:
                            d_p = p.grad
                    else:
                        noise = torch.randn(*p.grad.shape)
                        if self.scale_noise:
                            ps = p.grad.norm()
                            noise = ps*noise/noise.norm()
                        d_p = p.grad + self.grad_noise*noise
                else:
                    d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss


def rmsprop_prop(params: List[Tensor],
            grads: List[Tensor],
            square_avgs: List[Tensor],
            grad_avgs: List[Tensor],
            momentum_buffer_list: List[Tensor],
            *,
            lr: float,
            alpha: float,
            eps: float,
            weight_decay: float,
            momentum: float,
            centered: bool,
            prop=1):
    r"""Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """

    for i, param in enumerate(params):
        grad = grads[i]
        grad_original = grad.clone()
        square_avg = square_avgs[i]

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            grad_avg = grad_avgs[i]
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(eps)
        else:
            avg = square_avg.sqrt().add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            buf.mul_(momentum).addcdiv_(grad, avg)
            param.add_(buf, alpha=-lr*prop)
            param.add_(grad_original, alpha=-lr*(1-prop))
        else:
            param.addcdiv_(grad, avg, value=-lr*prop)
            param.add_(grad_original, alpha=-lr*(1-prop))



class NoisyRMSprop(torch.optim.RMSprop):
    r"""Implements Noisy RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, grad_noise=0, noise_p_idx=None):
        super().__init__(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False)
        self.grad_noise = grad_noise
        self.noise_p_idx = noise_p_idx

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for k, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')

                if self.grad_noise > 0.0:
                    if self.noise_p_idx is not None:
                        if k in self.noise_p_idx:
                            d_p = p.grad \
                                    + self.grad_noise*torch.randn(*p.grad.shape)
                        else:
                            d_p = p.grad
                    else:
                        d_p = p.grad \
                                + self.grad_noise*torch.randn(*p.grad.shape)
                else:
                    d_p = p.grad
                grads.append(d_p)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1


            F.rmsprop(params_with_grad, grads, square_avgs, grad_avgs,
                    momentum_buffer_list, group['lr'], group['alpha'],
                    group['eps'], group['weight_decay'], group['momentum'],
                    group['centered'])

        return loss

# class SGD_RMSprop_Comb(torch.optim.RMSprop):
    # r"""Implements a convex combination of RMSprop and SGD algorithm.

    # Proposed by G. Hinton in his
    # `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    # The centered version first appears in `Generating Sequences
    # With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    # The implementation here takes the square root of the gradient average before
    # adding epsilon (note that TensorFlow interchanges these two operations). The effective
    # learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    # is the scheduled learning rate and :math:`v` is the weighted moving average
    # of the squared gradient.

    # Args:
        # params (iterable): iterable of parameters to optimize or dicts defining
            # parameter groups
        # lr (float, optional): learning rate (default: 1e-2)
        # momentum (float, optional): momentum factor (default: 0)
        # alpha (float, optional): smoothing constant (default: 0.99)
        # eps (float, optional): term added to the denominator to improve
            # numerical stability (default: 1e-8)
        # centered (bool, optional) : if ``True``, compute the centered RMSProp,
            # the gradient is normalized by an estimation of its variance
        # weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    # """

    # def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 # momentum=0, centered=False, prop=1):
        # super().__init__(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 # momentum=0, centered=False)
        # self.prop = prop

    # @torch.no_grad()
    # def step(self, closure=None):
        # """Performs a single optimization step.

        # Args:
            # closure (callable, optional): A closure that reevaluates the model
                # and returns the loss.
        # """
        # loss = None
        # if closure is not None:
            # with torch.enable_grad():
                # loss = closure()

        # for group in self.param_groups:
            # params_with_grad = []
            # grads = []
            # square_avgs = []
            # grad_avgs = []
            # momentum_buffer_list = []

            # for k, p in enumerate(group['params']):
                # if p.grad is None:
                    # continue
                # params_with_grad.append(p)

                # if p.grad.is_sparse:
                    # raise RuntimeError('RMSprop does not support sparse gradients')

                # d_p = p.grad
                # grads.append(d_p)

                # state = self.state[p]

                # # State initialization
                # if len(state) == 0:
                    # state['step'] = 0
                    # state['square_avg'] = torch.zeros_like(
                        # p, memory_format=torch.preserve_format)
                    # if group['momentum'] > 0:
                        # state['momentum_buffer'] = torch.zeros_like(
                            # p, memory_format=torch.preserve_format)
                    # if group['centered']:
                        # state['grad_avg'] = torch.zeros_like(
                            # p, memory_format=torch.preserve_format)

                # square_avgs.append(state['square_avg'])

                # if group['momentum'] > 0:
                    # momentum_buffer_list.append(state['momentum_buffer'])
                # if group['centered']:
                    # grad_avgs.append(state['grad_avg'])

                # state['step'] += 1


            # rmsprop_prop(params_with_grad, grads, square_avgs, grad_avgs,
                         # momentum_buffer_list, lr=group['lr'],
                         # alpha=group['alpha'], eps=group['eps'],
                         # weight_decay=group['weight_decay'],
                         # momentum=group['momentum'],
                         # centered=group['centered'], prop=self.prop)

        # return loss


class SGD_RMSprop_Comb(Optimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, prop=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(SGD_RMSprop_Comb, self).__init__(params, defaults)

        self.prop = prop

    def __setstate__(self, state):
        super(SGD_RMSprop_Comb, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1


            F.rmsprop(params_with_grad,
                      grads,
                      square_avgs,
                      grad_avgs,
                      momentum_buffer_list,
                      lr=group['lr'],
                      alpha=group['alpha'],
                      eps=group['eps'],
                      weight_decay=group['weight_decay'],
                      momentum=group['momentum'],
                      centered=group['centered'],
                      # prop=self.prop
                     )

        return loss


class RMSprop(Optimizer):
    r"""Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, prop=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)
        self.prop = prop

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)

                if p.grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avgs.append(state['square_avg'])

                if group['momentum'] > 0:
                    momentum_buffer_list.append(state['momentum_buffer'])
                if group['centered']:
                    grad_avgs.append(state['grad_avg'])

                state['step'] += 1


            rmsprop_prop(params_with_grad,
                      grads,
                      square_avgs,
                      grad_avgs,
                      momentum_buffer_list,
                      lr=group['lr'],
                      alpha=group['alpha'],
                      eps=group['eps'],
                      weight_decay=group['weight_decay'],
                      momentum=group['momentum'],
                      centered=group['centered'],
                        prop=self.prop)

        return loss
