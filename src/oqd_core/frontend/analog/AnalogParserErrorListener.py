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

from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import RecognitionException
from oqd_compiler_infrastructure.error import DefaultErrorCollector


class GenericAnalogParserError(RecognitionException): ...


class AnalogParserErrorListener(ErrorListener, DefaultErrorCollector):
    def error_message_formatter(self, idx: int, error: Exception) -> str:
        return f"({idx}) {error.__class__.__name__}: {error.args[0]}"

    def report_errors(self, *args, **kwargs):
        if self:
            raise self._error_class(self.__repr__())

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        e = (
            GenericAnalogParserError(
                message=msg,
                recognizer=recognizer,
                input=recognizer._input,
                ctx=recognizer._ctx,
            )
            if e is None
            else e
        )
        msg = "" if msg is None else msg
        err = (
            f"({recognizer.__class__.__name__}) "
            + "line "
            + str(line)
            + ":"
            + str(column)
            + " "
            + msg
        )
        e.message = err
        e.args = (err, *e.args[1:])
        self.append(e)
