# Verbatim

Video in, accurate speaker labelled transcript out. It does that one job and nothing else.

```
pip install -e .
verbatim models --download
verbatim transcribe meeting.mp4 -f txt,srt,json
verbatim serve        # local web app at http://127.0.0.1:8765
```

Work in progress. Full documentation and measured results are coming in the next commit.
