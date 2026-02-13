# theme_word_generation_task.py
# PsychoPy equivalent of Lab.js "enter a word, press Enter after each word"

from psychopy import visual, event, core
import os, csv
from datetime import datetime

current_directory = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(current_directory, "..", "theme_words_data")


def show_instructions(win):
    page1 = (
        "Theme Word Generation\n\n"
        "Please read the following instructions carefully.\n\n"
        "You will be asked to freely generate up to 10 words that best describe the central themes and ideas of the episodes you just viewed.\n\n"
        "Enter ONE word at a time and press ENTER to submit.\n\n"
        "Press ENTER to start."
    )

    stim1 = visual.TextStim(
        win, text=page1, wrapWidth=1300, color="black",
        height=40, units="pix"
    )
    stim1.draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    win.flip()


def run_theme_words(win, participant_id, max_words=10):
    """
    Collect up to max_words theme words.
    - Enter submits the current word (must be non-empty)
    - 0 finishes early (only when current input is empty, to avoid accidental finishes)
    - Backspace edits
    Returns: list of (word_index, word, rt_seconds)
    """
    responses = []

    # UI elements
    prompt_text = visual.TextStim(
        win,
        text='Type a theme word and press Enter.',
        height=0.05,
        color="gray",
        pos=(0, 0.15)
    )

    box_outline = visual.Rect(
        win,
        width=0.8,
        height=0.15,
        lineColor="gray",
        fillColor=None,
        pos=(0, -0.05),
        lineWidth=3
    )

    input_text = visual.TextStim(
        win,
        text="",
        height=0.08,
        color="black",
        pos=(0, -0.05)
    )

    counter_text = visual.TextStim(
        win,
        text="",
        height=0.05,
        color="black",
        pos=(0, -0.25)
    )

    # clear any buffered keypresses before starting
    event.clearEvents()

    typed_word = ""
    response_clock = core.Clock()
    response_clock.reset()

    while True:
        # Stop if reached quota
        if len(responses) >= max_words:
            break

        # Draw
        counter_text.text = f"{len(responses)}/{max_words}"
        input_text.text = typed_word

        prompt_text.draw()
        box_outline.draw()
        input_text.draw()
        counter_text.draw()
        win.flip()

        keys = event.getKeys(timeStamped=response_clock)

        for key, t in keys:
            if key in ["escape"]:
                win.close()
                core.quit()

            # Finish key: "0"
            elif key == "0":
                # only finish if nothing currently typed
                if typed_word.strip() == "":
                    return responses
                else:
                    typed_word += "0"

            # Submit key: Enter
            elif key == "return":
                entry = typed_word.strip()
                if entry == "":
                    continue

                # enforce single word (optional, but matches your instructions)
                entry = entry.split()[0]

                responses.append((len(responses) + 1, entry, t))
                typed_word = ""

                # reset RT timer for next entry so RT is "time to submit this word"
                response_clock.reset()

            elif key == "backspace":
                typed_word = typed_word[:-1]

            else:
                # regular character keys (letters, numbers, etc.)
                if len(key) == 1:
                    typed_word += key

    return responses


def save_theme_words(responses, participant_id, label=""):
    """
    Save to CSV in SAVE_DIR.
    responses: list of (word_index, word, rt_seconds)
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{participant_id}_{label}_themeWords_{current_datetime}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ParticipantID", "WordIndex", "ThemeWord", "RT (s)"])
        for word_index, word, rt in responses:
            writer.writerow([participant_id, word_index, word, rt])

    print(f"[SAVED] Theme words saved to {filepath}")
    return filepath


if __name__ == "__main__":
    win = visual.Window(size=(1440, 960), color="white", fullscr=False)
    show_instructions(win)
    data = run_theme_words(win, participant_id="PTEST", max_words=10)
    save_theme_words(data, participant_id="PTEST", label="TEST")
    win.close()
    core.quit()
