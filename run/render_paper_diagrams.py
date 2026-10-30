"""Rebuild the paper's quantum diagrams and trained cross illustration on CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def compile_tex(source: Path, build: Path, latex: str) -> Path:
    build.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [latex, '-halt-on-error', '-interaction=nonstopmode',
         f'-output-directory={build}', source.name],
        cwd=source.parent, capture_output=True, text=True,
        env={**os.environ, 'SOURCE_DATE_EPOCH': '0'},
    )
    (build / f'{source.stem}.stdout').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{source.name} failed; see {build / (source.stem + ".stdout")}')
    return build / (source.stem + '.pdf')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'figs')
    parser.add_argument('--latex', default='pdflatex')
    args = parser.parse_args(argv)
    if shutil.which(args.latex) is None:
        raise SystemExit('pdfLaTeX is required; see REPRODUCING.md for the TeX packages.')
    output = args.output_dir.resolve()
    (output / 'static').mkdir(parents=True, exist_ok=True)
    crosses = ROOT / 'figsrc/figure1'
    subprocess.run([sys.executable, str(crosses / 'render_crosses.py'),
                    '--prepared-root', str(ROOT / 'results/figure1_crosses')],
                   check=True, cwd=ROOT,
                   env={**os.environ, 'SOURCE_DATE_EPOCH': '0', 'MPLBACKEND': 'Agg'})
    for name in ('crosses_full_pcs', 'crosses_full_qcs', 'crosses_architecture',
                 'crosses_pcs_circuit', 'crosses_qcs_circuit', 'crosses'):
        compiled = compile_tex(crosses / f'{name}.tex', crosses / 'build', args.latex)
        if name in ('crosses', 'crosses_architecture'):
            shutil.copyfile(compiled, output / 'static' / f'figure1_{name}.pdf')
    for source in sorted((ROOT / 'figsrc/quantikz').glob('*.tex')):
        compiled = compile_tex(source, ROOT / 'build/diagrams', args.latex)
        target = output / ('static' if source.stem == 'local_pcs_synthesis' else '') / compiled.name
        shutil.copyfile(compiled, target)
    # This vector drawing is the original standalone asset; no editable source
    # is required to reproduce it in the paper.
    shutil.copyfile(ROOT / 'figsrc/assets/multiplexer.pdf', output / 'static/multiplexer.pdf')
    print(f'Paper diagrams written to {output}')


if __name__ == '__main__':
    main()
