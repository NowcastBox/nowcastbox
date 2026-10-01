"""Visual themes shared by the Plotly and Matplotlib backends.

A :class:`Theme` bundles every visual decision of a chart (categorical palette, the
fixed colours of the *observed* / *in-sample* / *out-of-sample* roles, the diverging
pair used for signed quantities such as loadings and news impacts, fonts, surfaces and
grid). Every plotting function of :mod:`nowcastbox.visualization` accepts
``theme=`` (a :class:`Theme` or the name of a registered one); ``None`` uses the
current default (see :func:`set_default_theme`).

Built-in themes
---------------
``"default"``
    Light surface, categorical palette validated for colour-vision deficiencies
    (fixed slot order, never cycled), recessive grid.
``"academic"``
    Publication theme: white surface, serif font, dark-grey and muted colours that
    survive greyscale printing.
``"presentation"``
    Larger fonts and thicker lines for slides.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "Theme",
    "get_theme",
    "list_themes",
    "register_theme",
    "set_default_theme",
]

# Categorical palette: fixed slot order chosen so that adjacent slots stay separable
# under the common colour-vision deficiencies (the order is part of the design).
_CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)


@dataclass(frozen=True)
class Theme:
    """Visual configuration of a chart, shared by both backends.

    Parameters
    ----------
    name : str
        Theme identifier.
    colors : tuple of str
        Categorical palette (hex), used in this fixed order for series identity.
        A series beyond the palette length reuses the colours (consider folding them
        into an "Other" group instead).
    observed_color, in_sample_color, out_of_sample_color : str
        Colours of the three roles of a nowcast chart.
    interval_color : str
        Colour of prediction-interval bands and fan charts (drawn with transparency).
    positive_color, negative_color : str
        Diverging poles for signed quantities (news impacts, loadings).
    neutral_color : str
        Diverging midpoint and colour of totals in waterfalls.
    font_family : str
        Font family (CSS font stack for Plotly; first family is used by Matplotlib).
    font_size : int
        Base font size in points.
    title_size : int
        Title font size in points.
    text_color, muted_text_color : str
        Primary and secondary ink (labels never take a series colour).
    background_color, plot_background_color : str
        Figure and plotting-area surfaces.
    grid_color : str
        Grid-line colour (kept recessive).
    line_width : float
        Width of data lines in points/pixels.
    marker_size : float
        Marker size (Plotly pixels; Matplotlib uses ``marker_size / 1.5`` points).
    figsize : tuple of float
        Default Matplotlib figure size in inches.
    plotly_template : str
        Base Plotly template.
    width, height : int
        Default Plotly figure size in pixels (``width=0`` lets Plotly autosize).

    Examples
    --------
    >>> from nowcastbox.visualization.themes import Theme, get_theme
    >>> base = get_theme("default")
    >>> custom = base.replace(name="corporate", colors=("#003366", "#ff6600"))
    >>> custom.color(3)
    '#ff6600'
    """

    name: str
    colors: tuple[str, ...] = _CATEGORICAL
    observed_color: str = "#0b0b0b"
    in_sample_color: str = "#2a78d6"
    out_of_sample_color: str = "#eb6834"
    interval_color: str = "#eb6834"
    positive_color: str = "#2a78d6"
    negative_color: str = "#e34948"
    neutral_color: str = "#8a8984"
    font_family: str = "Inter, Helvetica Neue, Arial, sans-serif"
    font_size: int = 12
    title_size: int = 15
    text_color: str = "#0b0b0b"
    muted_text_color: str = "#52514e"
    background_color: str = "#ffffff"
    plot_background_color: str = "#fcfcfb"
    grid_color: str = "#e6e5e0"
    line_width: float = 2.0
    marker_size: float = 8.0
    figsize: tuple[float, float] = (9.0, 4.8)
    plotly_template: str = "plotly_white"
    width: int = 0
    height: int = 460
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Theme name must be a non-empty string.")
        if not self.colors:
            raise ValueError("Theme colors must contain at least one colour.")
        if self.font_size <= 0 or self.title_size <= 0:
            raise ValueError("Font sizes must be positive.")
        if self.line_width <= 0 or self.marker_size <= 0:
            raise ValueError("line_width and marker_size must be positive.")

    def color(self, index: int) -> str:
        """Return the categorical colour of slot ``index`` (0-based, wraps around).

        Parameters
        ----------
        index : int
            Slot number.

        Returns
        -------
        str
            Hex colour.

        Examples
        --------
        >>> get_theme("default").color(0)
        '#2a78d6'
        """
        return self.colors[index % len(self.colors)]

    def replace(self, **changes: Any) -> Theme:
        """Return a copy with some fields changed.

        Parameters
        ----------
        **changes
            Field values to change.

        Returns
        -------
        Theme
            New theme.

        Examples
        --------
        >>> get_theme("default").replace(font_size=14).font_size
        14
        """
        return replace(self, **changes)

    def matplotlib_rc(self) -> dict[str, Any]:
        """Matplotlib ``rcParams`` implementing the theme (use with ``rc_context``).

        Returns
        -------
        dict
            rcParams overrides.

        Examples
        --------
        >>> rc = get_theme("academic").matplotlib_rc()
        >>> rc["font.family"]
        'serif'
        """
        family = self.font_family.split(",")[0].strip()
        generic = {"serif", "sans-serif", "monospace"}
        rc: dict[str, Any] = {
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.font_size,
            "axes.edgecolor": self.grid_color,
            "axes.labelcolor": self.muted_text_color,
            "axes.facecolor": self.plot_background_color,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlecolor": self.text_color,
            "figure.facecolor": self.background_color,
            "grid.color": self.grid_color,
            "grid.linewidth": 0.8,
            "xtick.color": self.muted_text_color,
            "ytick.color": self.muted_text_color,
            "legend.frameon": False,
            "lines.linewidth": self.line_width,
            "savefig.facecolor": self.background_color,
        }
        if family in generic:
            rc["font.family"] = family
        else:
            rc["font.family"] = "sans-serif"
            rc["font.sans-serif"] = [family, "DejaVu Sans"]
        return rc

    def plotly_layout(self) -> dict[str, Any]:
        """Plotly layout settings implementing the theme.

        Returns
        -------
        dict
            Keyword arguments for ``Figure.update_layout``.

        Examples
        --------
        >>> get_theme("default").plotly_layout()["template"]
        'plotly_white'
        """
        layout: dict[str, Any] = {
            "template": self.plotly_template,
            "font": {
                "family": self.font_family,
                "size": self.font_size,
                "color": self.text_color,
            },
            "title": {"font": {"size": self.title_size, "color": self.text_color}},
            "paper_bgcolor": self.background_color,
            "plot_bgcolor": self.plot_background_color,
            "colorway": list(self.colors),
            "height": self.height,
            "hovermode": "x unified",
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
            "margin": {"l": 60, "r": 30, "t": 80, "b": 50},
        }
        if self.width > 0:
            layout["width"] = self.width
        return layout

    def plotly_axis(self) -> dict[str, Any]:
        """Plotly axis settings (recessive grid, muted ticks).

        Returns
        -------
        dict
            Keyword arguments for ``update_xaxes``/``update_yaxes``.

        Examples
        --------
        >>> get_theme("default").plotly_axis()["showgrid"]
        True
        """
        return {
            "showgrid": True,
            "gridcolor": self.grid_color,
            "zeroline": False,
            "linecolor": self.grid_color,
            "tickfont": {"color": self.muted_text_color},
        }


_THEMES: dict[str, Theme] = {
    "default": Theme(name="default"),
    "academic": Theme(
        name="academic",
        colors=(
            "#1f1f1f",
            "#4a6fa5",
            "#b5651d",
            "#5b8c5a",
            "#8c6bb1",
            "#7f7f7f",
            "#a23b72",
            "#2e8b8b",
        ),
        observed_color="#000000",
        in_sample_color="#4a6fa5",
        out_of_sample_color="#b5651d",
        interval_color="#7f7f7f",
        positive_color="#4a6fa5",
        negative_color="#b03a2e",
        neutral_color="#9a9a9a",
        font_family="serif",
        font_size=10,
        title_size=11,
        plot_background_color="#ffffff",
        grid_color="#ececec",
        line_width=1.4,
        marker_size=6.0,
        figsize=(6.5, 3.6),
        plotly_template="simple_white",
    ),
    "presentation": Theme(
        name="presentation",
        font_size=15,
        title_size=20,
        line_width=3.0,
        marker_size=10.0,
        figsize=(12.0, 6.0),
        height=560,
    ),
}
_DEFAULT: list[str] = ["default"]


def register_theme(theme: Theme, *, overwrite: bool = False) -> None:
    """Register a theme so it can be selected by name.

    Parameters
    ----------
    theme : Theme
        Theme to register under ``theme.name``.
    overwrite : bool, default False
        Allow replacing an existing theme.

    Raises
    ------
    TypeError
        If ``theme`` is not a :class:`Theme`.
    ValueError
        If the name is taken and ``overwrite`` is False.

    Examples
    --------
    >>> register_theme(get_theme("default").replace(name="doc_example"), overwrite=True)
    >>> "doc_example" in list_themes()
    True
    """
    if not isinstance(theme, Theme):
        raise TypeError(f"theme must be a Theme instance; got {type(theme).__name__}.")
    if theme.name in _THEMES and not overwrite:
        raise ValueError(f"Theme {theme.name!r} already exists; pass overwrite=True.")
    _THEMES[theme.name] = theme


def list_themes() -> list[str]:
    """Names of the registered themes.

    Returns
    -------
    list of str
        Sorted theme names.

    Examples
    --------
    >>> {"default", "academic", "presentation"} <= set(list_themes())
    True
    """
    return sorted(_THEMES)


def get_theme(theme: Theme | str | None = None) -> Theme:
    """Resolve a theme argument.

    Parameters
    ----------
    theme : Theme, str or None
        A theme, the name of a registered theme, or ``None`` for the current default.

    Returns
    -------
    Theme
        The resolved theme.

    Raises
    ------
    ValueError
        Unknown theme name.
    TypeError
        Argument of another type.

    Examples
    --------
    >>> get_theme().name
    'default'
    >>> get_theme("academic").font_family
    'serif'
    """
    if theme is None:
        return _THEMES[_DEFAULT[0]]
    if isinstance(theme, Theme):
        return theme
    if isinstance(theme, str):
        try:
            return _THEMES[theme]
        except KeyError:
            raise ValueError(f"Unknown theme {theme!r}; available: {list_themes()}.") from None
    raise TypeError(f"theme must be a Theme, a theme name or None; got {type(theme).__name__}.")


def set_default_theme(theme: Theme | str) -> Theme:
    """Set the theme used when plotting functions receive ``theme=None``.

    Parameters
    ----------
    theme : Theme or str
        Theme (registered automatically if new) or registered name.

    Returns
    -------
    Theme
        The previous default theme (handy to restore it).

    Examples
    --------
    >>> previous = set_default_theme("academic")
    >>> get_theme().name
    'academic'
    >>> _ = set_default_theme(previous)
    """
    previous = get_theme()
    resolved = get_theme(theme)
    if isinstance(theme, Theme) and theme.name not in _THEMES:
        register_theme(theme)
    _THEMES[resolved.name] = resolved
    _DEFAULT[0] = resolved.name
    return previous
