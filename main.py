from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager

import os
import threading
import time
import requests

from plyer import filechooser

default_directory = os.path.abspath(os.curdir)
url = 'https://app.nanonets.com/api/v2/ImageCategorization/LabelFile/'


class MainWindow(Screen):
    def identify(self):
        input_path = filechooser.open_file(title="Select an Image")[0]
        data = {'file': open(input_path, 'rb'), 'modelId': ('', '46d349af-85c1-46ee-bc70-330da2a736ee')}
        response = requests.post(url, auth=requests.auth.HTTPBasicAuth('5Ner4sKSl0ikoPQ074_35YoNk9JBX__W', ''),
                                 files=data)
        a = response.text.replace(
            "{\"message\":\"Success\",\"result\":[{\"message\":\"Success\",\"prediction\":[{\"label\":\"", "").split(
            "\"")
        label = a[0].replace("_", " ").title()
        self.ans.text = "Eye Defect - " + label

    def identify_url(self, url_given):
        if url_given:
            headers = {'accept': 'application/x-www-form-urlencoded'}
            data = {'modelId': '46d349af-85c1-46ee-bc70-330da2a736ee', 'urls': [self.url_entry.text]}
            response = requests.request('POST', url, headers=headers,
                                        auth=requests.auth.HTTPBasicAuth('5Ner4sKSl0ikoPQ074_35YoNk9JBX__W', ''),
                                        data=data)
            a = response.text.replace(
                "{\"message\":\"Success\",\"result\":[{\"message\":\"Success\",\"prediction\":[{\"label\":\"",
                "").split("\"")
            label = a[0].replace("_", " ").title()
            self.ans.text = "Eye Defect - " + label


class SplashScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.get_screen)

    def get_screen(self, a):
        threading.Thread(target=self.change_screen).start()

    @mainthread
    def change_screen(self):
        time.sleep(2)
        self.manager.current = "Main"


class WindowManager(ScreenManager):
    pass


class Gui(App):
    def build(self):
        return Builder.load_file("gui.kv")


if __name__ == '__main__':
    Gui().run()
