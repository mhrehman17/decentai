class ResourceManager:
    def __init__(self):
        self.available_resources = {"default": {"gpu": True}}

    def find_resource(self, requirements):
        return "default"  # For simplicity, always return the default resource