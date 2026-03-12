from .AnalogLexer import AnalogLexer
from .AnalogParser import AnalogParser
from .AnalogCircuitAST import _AnalogASTBuilder
from pathlib import Path
import antlr4
import typer

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
    tree = ast.model_dump_json(indent=2)

    with open(output_file, 'w') as f:
        f.write(tree)

if __name__ == '__main__':
    app()
    
