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
from .deserialize import deserialize_analog
from pathlib import Path
import antlr4
import typer

########################################################################################

app = typer.Typer()

@app.command()
def main(
    input_file: Path = typer.Option(None, "-i", "--input", help="Input file"),
    output_file: Path = typer.Option(None, "-o", "--output", help="Output file"),
    deserialize: bool = typer.Option(False, "-d", "--deserialize", help="Convert .ast to .analog"),
):
    
    with open(input_file, encoding="utf-8") as f:
        source = f.read()
        
    if deserialize:
        code = deserialize_analog(source)
        with open(output_file, 'w') as f:
            f.write(code)
        return
        
    stream = antlr4.InputStream(source)
    lexer = AnalogLexer(stream)
    parser = AnalogParser(antlr4.CommonTokenStream(lexer))
    tree = parser.program()
    builder = _AnalogASTBuilder()
    
    ast = builder.visit(tree)
    tree = ast.model_dump_json(indent=2, serialize_as_any=True)

    with open(output_file, 'w') as f:
        f.write(tree)

if __name__ == '__main__':
    app()
    
