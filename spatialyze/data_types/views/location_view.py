from .view import View


class LocationView(View):
    location = "trajBbox"
    timestamp = "timestamp"
    table_name = "General_Bbox"

    def __init__(self):
        super().__init__(self.table_name)
        self.default = True
