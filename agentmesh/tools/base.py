class ToolBase:
    name: str

    def call(self, **kwargs):
        raise NotImplementedError()
