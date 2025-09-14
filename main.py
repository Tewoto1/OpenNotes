from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from index import Index, tracked_file_types, workspace_dir
import difflib
import os

index = Index()
index.build_user_index()

class Handler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and any(event.src_path.endswith(ext) for ext in tracked_file_types):
            index.update_file_index(event.src_path)


def start_watcher(path="."):
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path=path, recursive=True)
    observer.start()
    return observer


def query(msg):
    pass

def interactive_loop():
    while True:
        try:
            cmd = input("> ").split()
            if not cmd:
                continue
            if cmd[0].strip() == 'quit' or cmd[0].strip() == 'q':
                break
            elif cmd[0].strip() == 'help' or cmd[0].strip() == 'h':
                print("Available commands: index, query, help, quit")
            elif cmd[0].strip() == 'index' or cmd[0].strip() == 'i':
                index.build_user_index()
            elif cmd[0].strip() == 'query' or cmd[0].strip() == 'q':
                if len(cmd) < 2:
                    print("Usage: query <message>")
                else:
                    msg = " ".join(cmd[1:])
                    query(msg)
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    observer = start_watcher()  # background watcher thread
    try:
        interactive_loop()  # foreground REPL
    finally:
        observer.stop()
        observer.join()

