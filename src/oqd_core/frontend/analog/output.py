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

from .AnalogLexer import AnalogLexer
from .AnalogParser import AnalogParser
from .AnalogCircuitAST import _AnalogASTBuilder
from pathlib import Path
import antlr4
import typer

from oqd_core.compiler.analog.passes.resolve import resolve_analog_circuit
from oqd_core.compiler.analog.passes.canonicalize import analog_operator_canonicalization

########################################################################################

app = typer.Typer()

@app.command()
def main(
    input_file: Path = typer.Option(None, "-i", "--input", help="Input .analog file"),
    output_file: Path = typer.Option("examples/analog/output.ast", "-o", "--output", help="Output file with parse tree"),
):
    
    with open(input_file, encoding="utf-8") as f:
        source = f.read()
        
    stream = antlr4.InputStream(source)
    lexer = AnalogLexer(stream)
    parser = AnalogParser(antlr4.CommonTokenStream(lexer))
    tree = parser.program()
    builder = _AnalogASTBuilder()
    
    ast = builder.visit(tree)
    ast = resolve_analog_circuit(ast)
    # ast = analog_operator_canonicalization(ast)
    tree = ast.model_dump_json(indent=2)

    with open(output_file, 'w') as f:
        f.write(tree)

if __name__ == '__main__':
    app()
    
