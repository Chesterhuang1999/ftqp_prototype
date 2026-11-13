"""Hand-written parser for FTQP .ft files."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from pathlib import Path
from typing import Any

from ftqp_prototype.parser.ast import (
    Allocation,
    Call,
    CallArg,
    Conditional,
    HardwareDecl,
    IdentifierArg,
    MapOp,
    Measurement,
    OperationDecl,
    Parallel,
    PauliLiteral,
    PrimitiveOp,
    Program,
    ProgramDecl,
    QECDecl,
    QubitRef,
    Reset,
    RepeatUntil,
    Sequence,
    Skip,
    SourceSpan,
    Statement,
    SubcircuitDecl,
    XorConst,
    as_sequence,
)
from ftqp_prototype.parser.lexer import Lexer, Token


class ParseError(Exception):
    """Raised when the input does not match the concrete .ft syntax."""


def parse_text(text: str, filename: str = "<string>") -> Program:
    return Parser(Lexer(text, filename).tokenize()).parse_program()


def parse_file(path: str | Path) -> Program:
    p = Path(path)
    return parse_text(p.read_text(encoding="utf-8"), str(p))


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.index = 0

    def parse_program(self) -> Program:
        decls = []
        span = self.current.span
        while not self.current.kind == "EOF":
            if self.match_keyword("hardware"):
                decls.append(self.parse_hardware(self.previous.span))
            elif self.match_keyword("qecc"):
                decls.append(self.parse_qecc(self.previous.span))
            elif self.match_keyword("def"):
                decls.append(self.parse_subcircuit(self.previous.span))
            elif self.match_keyword("program"):
                decls.append(self.parse_program_body(self.previous.span))
            else:
                self.error("expected top-level declaration: hardware, qecc, def, or program")
        return Program(span=span, declarations=decls)

    @property
    def current(self) -> Token:
        return self.tokens[self.index]

    @property
    def previous(self) -> Token:
        return self.tokens[self.index - 1]

    def advance(self) -> Token:
        tok = self.current
        if tok.kind != "EOF":
            self.index += 1
        return tok

    def check_symbol(self, value: str) -> bool:
        return self.current.is_symbol(value)

    def check_keyword(self, value: str) -> bool:
        return self.current.is_keyword(value)

    def match_symbol(self, value: str) -> bool:
        if self.check_symbol(value):
            self.advance()
            return True
        return False

    def match_keyword(self, value: str) -> bool:
        if self.check_keyword(value):
            self.advance()
            return True
        return False

    def expect_symbol(self, value: str) -> Token:
        if not self.match_symbol(value):
            self.error(f"expected symbol {value!r}")
        return self.previous

    def expect_keyword(self, value: str) -> Token:
        if not self.match_keyword(value):
            self.error(f"expected keyword {value!r}")
        return self.previous

    def expect_identifier(self, what: str = "identifier") -> Token:
        if self.current.kind != "IDENT":
            self.error(f"expected {what}")
        return self.advance()

    def match_declaration_end(self) -> None:
        self.match_symbol(";")

    def error(self, message: str) -> None:
        raise ParseError(f"{self.current.span.format()}: {message}; got {self.current.value!r}")

    def parse_hardware(self, span: SourceSpan) -> HardwareDecl:
        name = self.expect_identifier("hardware name").value
        self.match_symbol("=")
        obj = self.parse_object()
        self.match_declaration_end()
        params: dict[str, float] = {}
        for key, value in obj.items():
            normalized_key = _normalize_key(key)
            if isinstance(value, (int, float)):
                params[normalized_key] = float(value)
            elif isinstance(value, str):
                try:
                    params[normalized_key] = float(value)
                except ValueError as exc:
                    raise ParseError(
                        f"{span.format()}: hardware parameter {key!r} must be numeric"
                    ) from exc
            else:
                raise ParseError(f"{span.format()}: hardware parameter {key!r} must be numeric")
        return HardwareDecl(span=span, name=name, params=params)

    def parse_qecc(self, span: SourceSpan) -> QECDecl:
        name = self.expect_identifier("QEC scheme name").value
        hardware: str | None = None
        if self.match_keyword("using") or self.match_keyword("on"):
            hardware = self.expect_identifier("hardware configuration name").value
        self.match_symbol("=")
        raw = self.parse_object()
        self.match_declaration_end()
        return self._qec_from_object(name, hardware, raw, span)

    def _qec_from_object(
        self, name: str, hardware: str | None, raw: dict[str, Any], span: SourceSpan
    ) -> QECDecl:
        normalized = {_normalize_key(k): v for k, v in raw.items()}
        code = (
            normalized.get("[[n,k,d]]")
            or normalized.get("code")
            or normalized.get("params")
            or normalized.get("nkd")
        )
        if not (isinstance(code, list) and len(code) == 3):
            raise ParseError(f"{span.format()}: QEC scheme {name!r} requires [[n,k,d]]: [n,k,d]")
        n, k, d = [self._as_int(x, "[[n,k,d]]", span) for x in code]

        se = self._require_object(normalized, "se", span, "SE")
        prep = (
            self._require_object(normalized, "prep", span, "PREP")
            if "prep" in normalized
            else self._require_object(normalized, "preparation", span, "PREP")
        )

        logical_raw = normalized.get("opl") or normalized.get("op_l") or normalized.get("logical")
        faulty_raw = normalized.get("opf") or normalized.get("op_f") or normalized.get("faulty")
        logical_ops = self._parse_operation_map(logical_raw or {}, "logical", span)
        faulty_ops = self._parse_operation_map(faulty_raw or {}, "faulty", span)

        t_dec = self._as_expr(normalized.get("tdec") or normalized.get("t_dec") or "0")
        p_l = self._as_expr(normalized.get("pl") or normalized.get("p_l") or normalized.get("pL") or "0")
        n_aux_raw = normalized.get("naux") or normalized.get("n_aux")
        if n_aux_raw is None:
            raise ParseError(f"{span.format()}: QEC scheme {name!r} requires naux")
        n_aux = self._as_expr(n_aux_raw)

        return QECDecl(
            span=span,
            name=name,
            hardware=hardware,
            n=n,
            k=k,
            d=d,
            se_ancilla=self._field_expr(se, ["#ancilla", "ancilla", "nancilla", "n_ancilla"], span),
            t_se=self._field_expr(se, ["tse", "t_se", "time"], span),
            prep_time=self._field_expr(prep, ["time", "tprep", "t_prep"], span),
            prep_n_se=self._field_expr(prep, ["#se", "nse", "n_se"], span),
            prep_ancilla=self._field_expr(
                prep, ["#ancilla", "ancilla", "nancilla", "n_ancilla"], span
            ),
            logical_ops=logical_ops,
            faulty_ops=faulty_ops,
            t_dec=t_dec,
            p_l=p_l,
            n_aux=n_aux,
            raw=raw,
        )

    def _parse_operation_map(
        self, value: Any, kind: str, span: SourceSpan
    ) -> dict[str, OperationDecl]:
        if not isinstance(value, dict):
            raise ParseError(f"{span.format()}: {kind} operation list must be an object or pair list")
        result: dict[str, OperationDecl] = {}
        for op_name, op_raw in value.items():
            if not isinstance(op_raw, dict):
                raise ParseError(f"{span.format()}: operation {op_name!r} must have a specification")
            op = {_normalize_key(k): v for k, v in op_raw.items()}
            arity_value = op.get("arity")
            if arity_value is None:
                raise ParseError(f"{span.format()}: operation {op_name!r} is missing arity")
            arity = self._as_int(arity_value, f"{op_name}.arity", span)
            time = self._as_expr(op.get("time", "0"))
            ancilla = self._as_expr(
                op.get("#ancilla")
                or op.get("ancilla")
                or op.get("nancilla")
                or op.get("n_ancilla")
                or "0"
            )
            if kind == "logical":
                n_se = self._as_expr(op.get("#se") or op.get("nse") or op.get("n_se") or "0")
                result[op_name] = OperationDecl(
                    span=span,
                    name=op_name,
                    arity=arity,
                    time=time,
                    ancilla=ancilla,
                    n_se=n_se,
                    kind="logical",
                )
            else:
                p_error = self._as_expr(op.get("pf") or op.get("p_f") or op.get("perror") or "0")
                result[op_name] = OperationDecl(
                    span=span,
                    name=op_name,
                    arity=arity,
                    time=time,
                    ancilla=ancilla,
                    p_error=p_error,
                    kind="faulty",
                )
        return result

    def _require_object(
        self, obj: dict[str, Any], key: str, span: SourceSpan, display: str
    ) -> dict[str, Any]:
        value = obj.get(key)
        if not isinstance(value, dict):
            raise ParseError(f"{span.format()}: QEC scheme requires {display}: {{...}}")
        return {_normalize_key(k): v for k, v in value.items()}

    def _field_expr(self, obj: dict[str, Any], keys: SequenceABC[str], span: SourceSpan) -> str:
        for key in keys:
            nkey = _normalize_key(key)
            if nkey in obj:
                return self._as_expr(obj[nkey])
        raise ParseError(f"{span.format()}: missing required field, expected one of {', '.join(keys)}")

    def _as_int(self, value: Any, field: str, span: SourceSpan) -> int:
        if isinstance(value, bool):
            raise ParseError(f"{span.format()}: {field} must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                parsed = int(value)
            except ValueError as exc:
                raise ParseError(f"{span.format()}: {field} must be an integer") from exc
            return parsed
        raise ParseError(f"{span.format()}: {field} must be an integer")

    def _as_expr(self, value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value
        raise ParseError(f"{self.current.span.format()}: expected expression string or number")

    def parse_subcircuit(self, span: SourceSpan) -> SubcircuitDecl:
        name = self.expect_identifier("subcircuit name").value
        self.expect_symbol("(")
        params: list[str] = []
        param_types: dict[str, str] = {}
        if not self.check_symbol(")"):
            while True:
                if self.current.kind == "IDENT" and self._peek_symbol(":"):
                    param = self.advance().value
                    self.expect_symbol(":")
                    type_tok = self.expect_identifier("parameter type")
                    param_types[param] = _normalize_subcircuit_param_type(type_tok)
                else:
                    param = self.expect_identifier("parameter name").value
                params.append(param)
                if not self.match_symbol(","):
                    break
        self.expect_symbol(")")
        body = self.parse_braced_statement()
        self.match_declaration_end()
        return SubcircuitDecl(
            span=span, name=name, params=params, param_types=param_types, body=body
        )

    def parse_program_body(self, span: SourceSpan) -> ProgramDecl:
        body = self.parse_braced_statement()
        self.match_declaration_end()
        return ProgramDecl(span=span, body=body)

    def parse_braced_statement(self) -> Statement:
        start = self.expect_symbol("{").span
        body = self.parse_statement_sequence(stop_symbols={"}"})
        self.expect_symbol("}")
        if body.span is None:
            body.span = start
        return body

    def parse_parenthesized_statement(self) -> Statement:
        start = self.expect_symbol("(").span
        body = self.parse_statement_sequence(stop_symbols={")"})
        self.expect_symbol(")")
        if body.span is None:
            body.span = start
        return body

    def parse_statement_sequence(
        self,
        stop_symbols: set[str] | None = None,
        stop_keywords: set[str] | None = None,
    ) -> Statement:
        stop_symbols = stop_symbols or set()
        stop_keywords = stop_keywords or set()
        statements: list[Statement] = []
        while not self._at_stop(stop_symbols, stop_keywords):
            statements.append(self.parse_statement_expr())
            if self.match_symbol(";"):
                while self.match_symbol(";"):
                    pass
                continue
            if self._at_stop(stop_symbols, stop_keywords):
                break
            self.error("expected ';' between statements")
        return as_sequence(statements, statements[0].span if statements else self.current.span)

    def _at_stop(self, symbols: set[str], keywords: set[str]) -> bool:
        return (
            self.current.kind == "EOF"
            or (self.current.kind == "SYMBOL" and self.current.value in symbols)
            or (self.current.kind == "IDENT" and self.current.value in keywords)
        )

    def parse_statement_expr(self) -> Statement:
        left = self.parse_primary_statement()
        while self.match_symbol("||") or self.match_keyword("parallel"):
            op_span = self.previous.span
            right = self.parse_primary_statement()
            left = Parallel(span=op_span, left=left, right=right)
        return left

    def parse_primary_statement(self) -> Statement:
        if self.check_symbol("{"):
            return self.parse_braced_statement()
        if self.check_symbol("("):
            return self.parse_parenthesized_statement()
        if self.match_keyword("skip"):
            return Skip(span=self.previous.span)
        if self.match_keyword("prepare") or self.match_keyword("reset"):
            return self.parse_reset(self.previous.value, self.previous.span)
        if self.match_keyword("logical"):
            return self.parse_allocation(self.previous.span)
        if self.match_keyword("if"):
            return self.parse_conditional(self.previous.span)
        if self.match_keyword("repeat"):
            return self.parse_repeat_until(self.previous.span)
        if self.match_keyword("call"):
            return self.parse_call_statement(self.previous.span)
        if self.match_keyword("map"):
            return self.parse_map_op(self.previous.span)
        if self.match_keyword("xor"):
            return self.parse_xor_const(self.previous.span)
        if self.check_keyword("release"):
            self.error("release(q) is only valid as the closing delimiter of paper-style allocation")
        if self.current.kind == "IDENT" and self._peek_symbol("["):
            return self.parse_primitive_op()
        self.error("expected statement")
        raise AssertionError("unreachable")

    def parse_reset(self, keyword: str, span: SourceSpan) -> Reset:
        self.expect_symbol("(")
        target = self.parse_qref()
        self.expect_symbol(")")
        return Reset(span=span, target=target, keyword=keyword)

    def parse_allocation(self, span: SourceSpan) -> Allocation:
        name = self.expect_identifier("register name").value
        count = 1
        if self.match_symbol("["):
            count_tok = self.current
            if count_tok.kind != "NUMBER":
                self.error("allocation block count must be an integer literal")
            count = int(float(self.advance().value))
            self.expect_symbol("]")
        if not (self.match_symbol(":=") or self.match_symbol("=")):
            self.error("expected ':=' or '=' in allocation")
        self.expect_keyword("new")
        scheme = self.expect_identifier("QEC scheme name").value
        if self.check_symbol("{"):
            body = self.parse_braced_statement()
            return Allocation(span=span, register=name, count=count, scheme=scheme, body=body)
        self.expect_symbol(";")
        body = self.parse_statement_sequence_until_release(name)
        return Allocation(span=span, register=name, count=count, scheme=scheme, body=body)

    def parse_statement_sequence_until_release(self, register: str) -> Statement:
        statements: list[Statement] = []
        while True:
            if self.current.kind == "EOF":
                self.error(f"expected release({register}) before end of file")
            if self.check_keyword("release"):
                release_span = self.current.span
                self.advance()
                self.expect_symbol("(")
                actual = self.expect_identifier("released register name").value
                self.expect_symbol(")")
                if actual != register:
                    raise ParseError(
                        f"{release_span.format()}: expected release({register}), got release({actual})"
                    )
                return as_sequence(statements, release_span)
            statements.append(self.parse_statement_expr())
            if self.match_symbol(";"):
                continue
            self.error("expected ';' or release(...) in allocation body")

    def parse_conditional(self, span: SourceSpan) -> Conditional:
        measurement, targets = self.parse_measurement()
        self.expect_symbol(";")
        if self.check_symbol("{"):
            during = self.parse_braced_statement()
        elif self.check_symbol("("):
            during = self.parse_parenthesized_statement()
        else:
            during = self.parse_statement_sequence(stop_keywords={"then"})
        self.expect_keyword("then")
        if self.check_symbol("{"):
            then_branch = self.parse_braced_statement()
        elif self.check_symbol("("):
            then_branch = self.parse_parenthesized_statement()
        else:
            then_branch = self.parse_statement_expr()
        return Conditional(
            span=span,
            measurement=measurement,
            targets=targets,
            during=during,
            then_branch=then_branch,
        )

    def parse_repeat_until(self, span: SourceSpan) -> RepeatUntil:
        if not self.check_symbol("{"):
            self.error("repeat-until bodies must be brace-delimited")
        body = self.parse_braced_statement()
        self.expect_keyword("until")
        threshold_tok = self.current
        if threshold_tok.kind != "NUMBER":
            self.error("repeat-until threshold must be a numeric literal")
        self.advance()
        try:
            threshold = float(threshold_tok.value)
        except ValueError as exc:
            raise ParseError(f"{threshold_tok.span.format()}: invalid repeat-until threshold") from exc
        if threshold <= 0.0:
            raise ParseError(
                f"{threshold_tok.span.format()}: repeat-until threshold must be positive"
            )
        return RepeatUntil(span=span, body=body, threshold=threshold)

    def parse_measurement(self) -> tuple[Measurement, list[QubitRef]]:
        m_span = self.current.span
        self.expect_keyword("M")
        self.expect_symbol("(")
        was_string = False
        if self.current.kind == "STRING":
            name = self.advance().value
            was_string = True
        else:
            parts: list[str] = []
            while not self.check_symbol(")"):
                if self.current.kind not in {"IDENT", "SYMBOL", "NUMBER"}:
                    self.error("expected measurement name or Pauli string")
                parts.append(self.advance().value)
            name = "".join(parts).strip()
        self.expect_symbol(")")
        self.expect_symbol("[")
        targets = self.parse_qref_list()
        self.expect_symbol("]")
        return Measurement(span=m_span, name=name, was_string=was_string), targets

    def parse_primitive_op(self) -> PrimitiveOp:
        name_tok = self.expect_identifier("operation name")
        self.expect_symbol("[")
        targets = self.parse_qref_list()
        self.expect_symbol("]")
        return PrimitiveOp(span=name_tok.span, name=name_tok.value, targets=targets)

    def parse_map_op(self, span: SourceSpan) -> MapOp:
        name = self.expect_identifier("mapped operation name").value
        self.expect_symbol("[")
        targets = self.parse_qref_list()
        self.expect_symbol("]")
        return MapOp(span=span, name=name, targets=targets)

    def parse_xor_const(self, span: SourceSpan) -> XorConst:
        mask_tok = self.current
        if mask_tok.kind != "STRING":
            self.error("xor mask must be a quoted hexadecimal string")
        self.advance()
        try:
            value = int(mask_tok.value, 16)
        except ValueError as exc:
            raise ParseError(
                f"{mask_tok.span.format()}: invalid hexadecimal xor mask"
            ) from exc
        self.expect_keyword("into")
        name_tok = self.expect_identifier("xor target register")
        self.expect_symbol("[")
        start_tok = self.current
        if start_tok.kind != "NUMBER":
            self.error("xor range start must be an integer literal")
        start = int(float(self.advance().value))
        self.expect_symbol(":")
        width_tok = self.current
        if width_tok.kind != "NUMBER":
            self.error("xor range width must be an integer literal")
        width = int(float(self.advance().value))
        self.expect_symbol("]")
        return XorConst(
            span=span,
            target=QubitRef(name_tok.value, block=start, span=name_tok.span),
            width=width,
            value=value,
        )

    def parse_call_statement(self, span: SourceSpan) -> Call:
        name = self.expect_identifier("subcircuit name").value
        self.expect_symbol("(")
        args: list[CallArg] = []
        if not self.check_symbol(")"):
            while True:
                args.append(self.parse_call_arg())
                if not self.match_symbol(","):
                    break
        self.expect_symbol(")")
        return Call(span=span, name=name, args=args)

    def parse_call_arg(self) -> CallArg:
        if self.current.kind == "STRING":
            tok = self.advance()
            return PauliLiteral(tok.value, tok.span)
        if self.current.kind == "IDENT":
            tok = self.advance()
            if self.check_symbol("["):
                return self.finish_qref(tok)
            if _is_pauli(tok.value):
                return PauliLiteral(tok.value, tok.span)
            return IdentifierArg(tok.value, tok.span)
        self.error("expected call argument")
        raise AssertionError("unreachable")

    def parse_qref_list(self) -> list[QubitRef]:
        refs: list[QubitRef] = []
        if not self.check_symbol("]"):
            while True:
                refs.append(self.parse_qref())
                if not self.match_symbol(","):
                    break
        return refs

    def parse_qref(self) -> QubitRef:
        tok = self.expect_identifier("qubit or register name")
        return self.finish_qref(tok)

    def finish_qref(self, tok: Token) -> QubitRef:
        block: int | None = None
        logical: int | None = None
        if self.match_symbol("["):
            block = self.parse_index_literal()
            self.expect_symbol("]")
            if self.match_symbol("["):
                logical = self.parse_index_literal()
                self.expect_symbol("]")
        return QubitRef(tok.value, block, logical, tok.span)

    def parse_index_literal(self) -> int:
        if self.current.kind != "NUMBER":
            self.error("indices must be integer literals")
        tok = self.advance()
        try:
            value = int(tok.value)
        except ValueError as exc:
            raise ParseError(f"{tok.span.format()}: index must be an integer") from exc
        if value < 0:
            raise ParseError(f"{tok.span.format()}: index must be non-negative")
        return value

    def _peek_symbol(self, value: str) -> bool:
        return self.index + 1 < len(self.tokens) and self.tokens[self.index + 1].is_symbol(value)

    def parse_object(self) -> dict[str, Any]:
        self.expect_symbol("{")
        result: dict[str, Any] = {}
        if not self.check_symbol("}"):
            while True:
                key = self.parse_object_key()
                self.expect_symbol(":")
                result[key] = self.parse_value()
                if not self.match_symbol(","):
                    break
                if self.check_symbol("}"):
                    break
        self.expect_symbol("}")
        return result

    def parse_object_key(self) -> str:
        if self.match_symbol("[["):
            parts: list[str] = []
            while not (self.check_symbol("]") and self._peek_symbol("]")):
                if self.current.kind == "EOF":
                    self.error("unterminated [[...]] key")
                parts.append(self.advance().value)
            self.expect_symbol("]")
            self.expect_symbol("]")
            return "[[" + "".join(parts).replace(" ", "") + "]]"
        if self.current.kind in {"IDENT", "STRING"}:
            return self.advance().value
        if self.current.kind == "SYMBOL" and self.current.value == "#":
            return self.advance().value
        self.error("expected object key")
        raise AssertionError("unreachable")

    def parse_value(self) -> Any:
        if self.current.kind == "STRING":
            return self.advance().value
        if self.current.kind == "NUMBER":
            value = self.advance().value
            return float(value) if any(ch in value for ch in ".eE") else int(value)
        if self.current.kind == "IDENT":
            return self.advance().value
        if self.check_symbol("{"):
            return self.parse_object()
        if self.check_symbol("["):
            return self.parse_array_or_pair_list()
        self.error("expected value")
        raise AssertionError("unreachable")

    def parse_array_or_pair_list(self) -> Any:
        self.expect_symbol("[")
        values: list[Any] = []
        pairs: dict[str, Any] = {}
        pair_mode: bool | None = None
        if not self.check_symbol("]"):
            while True:
                save = self.index
                key: str | None = None
                try:
                    if self.current.kind in {"IDENT", "STRING"} or self.check_symbol("[["):
                        key = self.parse_object_key()
                        if self.match_symbol(":"):
                            pair_mode = True if pair_mode is None else pair_mode
                            if pair_mode is False:
                                raise ParseError(
                                    f"{self.current.span.format()}: cannot mix array values and pairs"
                                )
                            pairs[key] = self.parse_value()
                        else:
                            self.index = save
                            pair_mode = False if pair_mode is None else pair_mode
                            if pair_mode is True:
                                raise ParseError(
                                    f"{self.current.span.format()}: cannot mix array values and pairs"
                                )
                            values.append(self.parse_value())
                    else:
                        pair_mode = False if pair_mode is None else pair_mode
                        if pair_mode is True:
                            raise ParseError(
                                f"{self.current.span.format()}: cannot mix array values and pairs"
                            )
                        values.append(self.parse_value())
                except ParseError:
                    self.index = save
                    raise
                if not self.match_symbol(","):
                    break
                if self.check_symbol("]"):
                    break
        self.expect_symbol("]")
        return pairs if pair_mode else values


def _normalize_key(key: str) -> str:
    return key.strip().replace(" ", "").replace("-", "_").lower()


def _is_pauli(value: str) -> bool:
    return bool(value) and all(ch in "IXYZ" for ch in value)


def _normalize_subcircuit_param_type(tok: Token) -> str:
    normalized = tok.value.strip().replace("_", "").replace("-", "").lower()
    if normalized == "pauli":
        return "pauli"
    if normalized in {"codeblock", "block"}:
        return "codeblock"
    raise ParseError(
        f"{tok.span.format()}: unknown subcircuit parameter type {tok.value!r}; "
        "expected Pauli or Codeblock"
    )
