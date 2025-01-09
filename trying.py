import speech_recognition as sr
from pydub import AudioSegment
import os

def create_subtitle_files(audio_file, text_file, output_srt, output_sbv):
    # Load the audio file
    audio = AudioSegment.from_file(audio_file)

    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Read the text file
    with open(text_file, 'r') as file:
        text = file.read().splitlines()

    # Initialize subtitle counters and time
    srt_content = ""
    sbv_content = ""
    subtitle_count = 1
    start_time = 0

    # Process each line of text
    for line in text:
        # Estimate duration based on number of words (adjust as needed)
        duration = len(line.split()) * 500  # Assume 500ms per word
        end_time = start_time + duration

        # Format time for SRT (HH:MM:SS,mmm)
        srt_start = '{:02d}:{:02d}:{:02d},{:03d}'.format(
            start_time // 3600000,
            (start_time % 3600000) // 60000,
            (start_time % 60000) // 1000,
            start_time % 1000
        )
        srt_end = '{:02d}:{:02d}:{:02d},{:03d}'.format(
            end_time // 3600000,
            (end_time % 3600000) // 60000,
            (end_time % 60000) // 1000,
            end_time % 1000
        )

        # Format time for SBV (H:MM:SS.mmm)
        sbv_start = '{:01d}:{:02d}:{:02d}.{:03d}'.format(
            start_time // 3600000,
            (start_time % 3600000) // 60000,
            (start_time % 60000) // 1000,
            start_time % 1000
        )
        sbv_end = '{:01d}:{:02d}:{:02d}.{:03d}'.format(
            end_time // 3600000,
            (end_time % 3600000) // 60000,
            (end_time % 60000) // 1000,
            end_time % 1000
        )

        # Add to SRT content
        srt_content += f"{subtitle_count}\n{srt_start} --> {srt_end}\n{line}\n\n"

        # Add to SBV content
        sbv_content += f"{sbv_start},{sbv_end}\n{line}\n\n"

        # Update for next subtitle
        start_time = end_time
        subtitle_count += 1

    # Write SRT file
    with open(output_srt, 'w') as file:
        file.write(srt_content)

    # Write SBV file
    with open(output_sbv, 'w') as file:
        file.write(sbv_content)

# Example usage
audio_file = "/content/generated_video (1).mp3"
text_file = "/content/all_slides_notes.txt"
output_srt = "output.srt"
output_sbv = "output.sbv"

create_subtitle_files(audio_file, text_file, output_srt, output_sbv)