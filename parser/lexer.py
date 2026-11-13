"""Lexer for the FTQP prototype language."""

from __future__ import annotations

from dataclasses import dataclass

from ftqp_prototype.parser.ast import SourceSpan


class LexError(Exception):
    """Raised for lexical errors."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    span: SourceSpan

    def is_symbol(self, value: str) -> bool:
        return self.kind == "SYMBOL" and self.value == value

    def is_keyword(self, value: str) -> bool:
        return self.kind == "IDENT" and self.value == value


KEYWORDS = {
    "hardware",
    "qecc",
    "using",
    "on",
    "def",
    "program",
    "logical",
    "new",
    "release",
    "skip",
    "prepare",
    "reset",
    "if",
    "then",
    "repeat",
    "until",
    "call",
    "map",
    "xor",
    "into",
    "parallel",
    "M",
}


class Lexer:
    def __init__(self, text: str, filename: str = "<string>") -> None:
        self.text = text
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.length = len(text)

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while True:
            self._skip_trivia()
            if self.pos >= self.length:
                tokens.append(Token("EOF", "", self._span()))
                return tokens

            ch = self._peek()
            span = self._span()
            if ch.isalpha() or ch == "_":
                tokens.append(Token("IDENT", self._read_identifier(), span))
            elif ch.isdigit() or (ch == "." and self._peek(1).isdigit()):
                tokens.append(Token("NUMBER", self._read_number(), span))
            elif ch == '"':
                tokens.append(Token("STRING", self._read_string(), span))
            else:
                tokens.append(Token("SYMBOL", self._read_symbol(), span))

    def _span(self) -> SourceSpan:
        return SourceSpan(self.filename, self.line, self.column)

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.text[idx] if idx < self.length else ""

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_trivia(self) -> None:
        while self.pos < self.length:
            ch = self._peek()
            if ch.isspace():
                self._advance()
                continue
            if ch == "/" and self._peek(1) == "/":
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue
            if ch == "/" and self._peek(1) == "*":
                self._advance()
                self._advance()
                while self.pos < self.length:
                    if self._peek() == "*" and self._peek(1) == "/":
                        self._advance()
                        self._advance()
                        break
                    self._advance()
                else:
                    raise LexError(f"{self._span().format()}: unterminated block comment")
                continue
            if ch == "#":
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue
            break

    def _read_identifier(self) -> str:
        out = []
        while self.pos < self.length:
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                out.append(self._advance())
            else:
                break
        return "".join(out)

    def _read_number(self) -> str:
        out = []
        saw_dot = False
        while self.pos < self.length:
            ch = self._peek()
            if ch.isdigit():
                out.append(self._advance())
            elif ch == "." and not saw_dot:
                saw_dot = True
                out.append(self._advance())
            else:
                break
        if self._peek() in {"e", "E"}:
            out.append(self._advance())
            if self._peek() in {"+", "-"}:
                out.append(self._advance())
            if not self._peek().isdigit():
                raise LexError(f"{self._span().format()}: malformed exponent in number")
            while self._peek().isdigit():
                out.append(self._advance())
        return "".join(out)

    def _read_string(self) -> str:
        self._advance()
        out = []
        while self.pos < self.length:
            ch = self._advance()
            if ch == '"':
                return "".join(out)
            if ch == "\\":
                nxt = self._advance()
                escapes = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
                out.append(escapes.get(nxt, nxt))
            else:
                out.append(ch)
        raise LexError(f"{self._span().format()}: unterminated string literal")

    def _read_symbol(self) -> str:
        two = self._peek() + self._peek(1)
        if two in {":=", "[[", "||", "->", "<=", ">=", "==", "!="}:
            self._advance()
            self._advance()
            return two
        ch = self._advance()
        if ch in "{}[]():;,=+-*/^":
            return ch
        if ch == "\u2225":
            return "||"
        raise LexError(f"{self._span().format()}: unexpected character {ch!r}")
