# -*- coding: utf-8 -*-


import logging
import time


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


logger = logging.getLogger('plugin_service')
llm_logger = logging.getLogger('plugin_service.llm')
