import os
from IPython.lib import passwd

#c = c  # pylint:disable=undefined-variable
c = get_config()
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = int(os.getenv('PORT', 8888))
c.NotebookApp.open_browser = False

# sets a password if PASSWORD is set in the environment
c.NotebookApp.token = 'densepose'
