import streamlit.components.v1 as components
import os

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "latex_renderer",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("latex_renderer", path=build_dir)


def latex_renderer(code: str, key=None):
    """
    Renders TikZ-cd LaTeX code in Streamlit.
    """
    _component_func(code=code, key=key)