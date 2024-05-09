# DLproject_game

DL project by Shiqi Wang, Yanfeiyun Wu, Bingnan Huo

[Devpost Page](https://devpost.com/software/online-detective)

**Introduction**

Our group is driven by a passion for video games and game development. We've noticed that game store pages often don't serve as the best advertisements for games—especially true for indie developers who lack the resources and support that larger studios possess. To assist these indie developers in their marketing efforts, we plan to develop a deep learning model that generates compelling game summaries and matching marketing visualizations. This model will be trained on datasets containing game descriptions, ensuring it captures the essence of what the game is about. Our goal is to enhance marketing materials and provide guidance to developers on focusing their resources effectively to highlight their games' key selling points.

This project primarily employs supervised learning techniques. The text generation component is a structured prediction task, where the model is trained to predict and generate the next token that seamlessly integrates into the ongoing sentence, based on the initial game description and preceding tokens. Subsequently, the model undertakes a classification task to determine the game genre based on the crafted description. Finally, the image generation phase also utilizes supervised learning; here, the model is tasked with creating a visualization that corresponds appropriately to the specified genre. This structured approach ensures that each stage of the workflow is informed by and builds upon the outputs of the preceding tasks, facilitating a cohesive and targeted content generation process.
