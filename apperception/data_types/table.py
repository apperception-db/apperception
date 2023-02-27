from dataclasses import dataclass


class Table:
    @dataclass
    class __FieldRef:
        table: "Table"
        field: "str"
        type: "str"

        def __post_init__(self):
            assert self.table.fields[self.field] == self.type

    def __init__(
        self,
        name: "str",
        pkey: "str | None" = None,
        fkey: "tuple[str, __FieldRef] | None" = None,
        **fields: "str",
    ):
        self.name = name
        self.fields = dict() if fields is None else fields

        self.pkey = pkey
        assert pkey is None or pkey in fields, [*fields.keys()]

        self.fkey = fkey
        if fkey is not None:
            rfield, ref = fkey
            t1 = self.fields[rfield]
            t2 = ref.type
            assert t1 == t2, f"type mismatch: {rfield}:{t1} != {ref.field}:{t2}"

    def __getattr__(self, __name: str) -> "__FieldRef":
        if __name in self.fields:
            return self.__FieldRef(self, __name, self.fields[__name])
        raise Exception(f"field '{__name}' not found in [{', '.join(self.fields)}]")


t = Table("t", a="test")
b = Table("b", a="test", fkey=("a", t.a))
