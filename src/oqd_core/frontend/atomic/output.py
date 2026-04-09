# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import typer

from .AtomicCircuitAST import parse_atomic

########################################################################################

app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Option(None, "-i", "--input", help="Input file"),
    output_file: Path = typer.Option(None, "-o", "--output", help="Output file"),
):
    with open(input_file, encoding="utf-8") as f:
        source = f.read()

    circuit = parse_atomic(source)
    tree = circuit.model_dump_json(indent=2, serialize_as_any=True)

    with open(output_file, "w") as f:
        f.write(tree)


if __name__ == "__main__":
    app()
