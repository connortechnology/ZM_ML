#!/usr/bin/env python3
"""
A script to be used with ZoneMinder filter system. The filtering system can pass some CLI args
to the script. This is the beggining of the exploratory work on how to integrate this.

 This has 2 modes.

 - config mode: you can use ZM ML like monitor config to only annotate certain objects
     - This allows for filtering by label/group/zone
 - all: annotate anything it finds in the event
"""