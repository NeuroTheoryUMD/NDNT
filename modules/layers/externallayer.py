
import torch.nn as nn

class ExternalLayer(nn.Module):
    """This is a dummy 'layer' for the Extenal network that gets filled in by the passed-in network."""

    def __init__(self, input_dims=None,
        num_filters=None,
        output_dims=None,
        **kwargs):
        """
        ExternalLayer: Dummy layer for the Extenal network that gets filled in by the passed-in network.

        Args:
            input_dims: tuple or list of ints, (num_channels, height, width, lags)
            num_filters: number of output filters
            output_dims: tuple or list of ints, (num_channels, height, width, lags)
            **kwargs: additional arguments to pass to NDNLayer
        """
        assert input_dims is not None, "ExternalLayer: input_dims must be specified."
        assert num_filters is not None, "ExternalLayer: num_filters must be specified."
        assert output_dims is not None, "ExternalLayer: output_dims must be specified."

        """
        Make module constructor and set some 'shell' values that might be queried later"""
        super().__init__()
        self.input_dims = input_dims
        self.num_filters = num_filters
        self.filter_dims = [0,0,0,0]  # setting in case its used somewhere later -- probably not....
        self.output_dims = output_dims
        self.reg = None
        # External network will be plugged in after the FFnetwork constructor that called this, so not done here.

    def forward(self, x):
        """
        Forward pass for ExternalLayer.

        Args:
            x: input tensor

        Returns:
            y: output tensor
        """
        y = self.external_network(x) 
        return y