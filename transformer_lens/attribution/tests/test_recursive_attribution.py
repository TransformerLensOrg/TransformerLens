# def post_init(self):
#     """Validation with Post Init."""
#     # Check parent component is set (where required)
#     if self.type != ComponentType.MODEL:
#         assert (
#             self.parent_component is not None
#         ), f"'parent_component' must be set when type is {self.type}"
#         assert (
#             self.position is not None
#         ), f"'position' must be set when type is {self.type}"

#     # Check the layer is set (where required)
#     if self.type not in {
#         ComponentType.MODEL,
#         ComponentType.EMBED,
#         ComponentType.POSITIONAL_EMBED,
#     }:
#         assert (
#             self.layer is not None
#         ), f"'layer' must be set when type is {self.type}"

#     # For attention heads, check the head ID is set
#     if self.type == ComponentType.ATTENTION:
#         assert self.head is not None, "'head' must be set when type is ATTENTION"

#     # For MLP neurons, check the neuron ID is set
#     if self.type == ComponentType.MLP_NEURON:
#         assert (
#             self.neuron is not None
#         ), "'neuron' must be set when type is MLP_NEURON"
#     else:
#         assert (
#             self.neuron is None
#         ), f"'neuron' must be None when type is {self.type}"
