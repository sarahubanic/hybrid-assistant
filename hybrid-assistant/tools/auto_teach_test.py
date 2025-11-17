import time
import tkinter as tk
import os
import sys

# Ensure repository root is on sys.path so `src` can be imported when running from tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection_gui import DetectionGUI


class Evt:
    def __init__(self, x, y, widget):
        self.x = x
        self.y = y
        self.widget = widget


def save_pending_crop(app):
    try:
        crop = app.pending_dialog_crop
        if crop is None:
            print('[AUTO] No pending crop available')
            return False
        outp = os.path.join(app.learning_dir, 'auto_test_crop.jpg')
        cv2 = __import__('cv2')
        cv2.imwrite(outp, crop)
        print(f'[AUTO] Saved pending crop to: {outp}')
        return True
    except Exception as e:
        print('[AUTO] Error saving crop:', e)
        return False


def main():
    root = tk.Tk()
    root.title('Auto Test Runner')
    app = DetectionGUI(root, 'Auto Test')

    # Start camera after a short delay to let GUI initialize
    def start_camera():
        print('[AUTO] Starting camera')
        app.toggle_camera()

    root.after(1000, start_camera)

    # Open Teach dialog once camera is running
    def open_teach():
        print('[AUTO] Opening Teach dialog')
        # Ensure llm attribute exists (start_learning checks it)
        app.llm = getattr(app, 'llm', None)
        app.start_learning()

        # wait until dialog canvas appears and has size
        def wait_canvas():
            w = getattr(app, 'dialog_canvas_widget', None)
            if not w:
                root.after(200, wait_canvas)
                return
            cw = w.winfo_width()
            ch = w.winfo_height()
            if cw < 10 or ch < 10:
                root.after(200, wait_canvas)
                return

            print(f'[AUTO] dialog canvas size: {cw}x{ch}')

            # Freeze frame programmatically
            try:
                app.dialog_toggle_freeze(freeze=True)
            except Exception as e:
                print('[AUTO] dialog_toggle_freeze error', e)

            # simulate rectangle draw in canvas coords (center-right area)
            x0 = int(cw * 0.65)
            y0 = int(ch * 0.2)
            x1 = int(cw * 0.85)
            y1 = int(ch * 0.8)

            ev0 = Evt(x0, y0, w)
            ev1 = Evt(x1, y1, w)
            try:
                app._on_canvas_press(ev0)
                root.update()
                app._on_canvas_drag(ev1)
                root.update()
                app._on_canvas_release(ev1)
                root.update()
                print('[AUTO] Performed programmatic rectangle draw')
            except Exception as e:
                print('[AUTO] Error during simulated draw:', e)

            # Give preview a moment to update then save crop
            def finalize():
                ok = save_pending_crop(app)
                print('[AUTO] finalize done, quitting in 1s')
                root.after(1000, root.destroy)

            root.after(800, finalize)

        root.after(200, wait_canvas)

    root.after(2500, open_teach)

    root.mainloop()


if __name__ == '__main__':
    main()
