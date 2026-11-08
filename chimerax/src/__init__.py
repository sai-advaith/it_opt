from chimerax.core.toolshed import BundleAPI


class _ITOptAPI(BundleAPI):
    api_version = 1

    @staticmethod
    def register_command(bi, ci, logger):
        from . import cmd
        cmd.register_commands(logger)

    @staticmethod
    def start_tool(session, bi, ti):
        from .tool import ITOptTool
        return ITOptTool.get_singleton(session)


bundle_api = _ITOptAPI()
