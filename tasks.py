from pathlib import Path

from invoke import task

basepath = "/Users/janbolts/qode/sbi-for-connectomics"
overleaf = "/Users/janbolts/qode/connectomics-inference-manuscript"

open_cmd = "open"

fig_names = {
    "1": "paper/fig1",
    "2": "paper/fig2",
    "3": "paper/fig3",
}


@task
def convertpngpdf(c, fig):
    _convertsvg2pdf(c, fig)
    _convertpdf2png(c, fig)


@task
def syncoverleaf(c, fig):
    convertpngpdf(c, fig)
    c.run(f"cp {basepath}/{fig_names[fig]}/fig/*.pdf {overleaf}/figs/")
    c.run(f"cp {basepath}/{fig_names[fig]}/fig/*.png {overleaf}/figs/")


########################################################################################
# Helpers
########################################################################################
@task
def _convertsvg2pdf(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_svg2pdf(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.svg")
    for path in pathlist:
        c.run(f"inkscape {str(path)} --export-pdf={str(path)[:-4]}.pdf")


@task
def _convertpdf2png(c, fig):
    if fig is None:
        for f in range(len(fig_names)):
            _convert_pdf2png(c, str(f + 1))
        return
    pathlist = Path(f"{basepath}/{fig_names[fig]}/fig/").glob("*.pdf")
    for path in pathlist:
        c.run(
            f'inkscape {str(path)} --export-png={str(path)[:-4]}.png -b "white" --export-dpi=250'
        )
