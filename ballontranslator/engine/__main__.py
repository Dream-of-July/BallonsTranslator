"""Keep protocol stdout isolated before importing the application."""
from .transport import Writer, isolate_stdout


def main() -> None:
    writer = Writer(isolate_stdout())
    from .bootstrap import bootstrap
    app, window = bootstrap()
    from .server import Server
    server = Server(app, window, writer)
    server.start_reader()
    app.exec_()
    server.dispose()


if __name__ == '__main__':
    main()
