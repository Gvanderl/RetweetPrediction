
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Gvanderl/RetweetPrediction">
  </a>

  <h3 align="center">RetweetPrediction</h3>

  <p align="center">
    Predicting the number of retweets of a tweet based on text and metadata
    <br />
    <a href="https://github.com/Gvanderl/RetweetPrediction/issues">Report Bug</a>
    ·
    <a href="https://github.com/Gvanderl/RetweetPrediction/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project  is related to this [Kaggle competition](https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/overview). 
The goal of this project is to predict the number of retweets that a given tweet will get based on its contents and metadata.

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

You will need to have python 3.7 installed on your machine, as well as the pipenv virtual environment manager, [here](https://pipenv-fork.readthedocs.io/en/latest/install.html) are the installation instructions. If you want to use your own, you can find the requirements in the Pipfile
You can download the dataset to train the model [here](https://www.kaggle.com/c/covid19-retweet-prediction-challenge-2020/data)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Gvanderl/RetweetPrediction.git
   ```
   
You can find and change the paths in the [config.py](config.py) file to suit your setup.

2. Install the packages
   ```sh
   pipenv install
   ```

3. Run the script
   ```sh
   pipenv shell
   python main.py
   ```

The inference results should be saved in ./outputs if you left the default in [config.py](config.py).


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Project Link: [https://github.com/Gvanderl/RetweetPrediction](https://github.com/Gvanderl/RetweetPrediction)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Gvanderl/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/Gvanderl/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Gvanderl/repo.svg?style=for-the-badge
[forks-url]: https://github.com/Gvanderl/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/Gvanderl/repo.svg?style=for-the-badge
[stars-url]: https://github.com/Gvanderl/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/Gvanderl/repo.svg?style=for-the-badge
[issues-url]: https://github.com/Gvanderl/repo/issues
[license-shield]: https://img.shields.io/github/license/Gvanderl/repo.svg?style=for-the-badge
[license-url]: https://github.com/Gvanderl/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/Gvanderl
