# refugee_vessel_detection
Machine Learning and Image Analysis (Sentinel-2 and Marine traffic APIs) to detect migrant vessels in distress to facilitate rescue efforts.

## Background
With the ongoing humanitarian crisis having reached another tragic low point a few weeks ago, when a refugee camp on the island of Lesbos in Greece burned to the ground, I thought it would be a good idea to dedicate my resources helping to solve a real-world issue and learn more about Machine Learning and the processing of sensory inputs from different sources.
A tragically common situation that migrants boarding vessels from the African or Turkish coast of the Mediterranean Sea are facing, is having the engines of their underequipped and overcrowded vessels destroyed or taken away by some involved Marine authority and afterwards being left stranded in the middle of the sea by themselves. This is obviously an alarming and life-threatening situation for the migrants on such a vessel.

## Scope of the Project
My approach to help improve this situation with digital means is the following:
	* Fetch publicly available Satellite Imagery ([ESA’s Sentinel-2 data](https://scihub.copernicus.eu/) is an obvious and free candidate).
	* Analyze the images using machine learning and common computer vision practices to programmatically detect potential immigrant vessels.
	* Fetch data from a Marine Tracking API (such as [Marine Traffic](https://www.marinetraffic.com/), not open-source, their response to my request for research use is pending) and compare it to the previously obtained results of the image analysis.
	* Vessels that have been identified in step 2 but not in step 3 and fit the characteristics of a migrant ship can be pointed out to be observed and checked upon.
  
## Broader Context of Project
First research indicates that work in this direction has already been done up to the point of step 2 (see U. Kanjir (2019): _"Detecting migrant vessels in the Mediterranean Sea: Using Sentinel-2 images to aid humanitarian actions"_, in Acta Astronautica 155, pp. 45-50). The result obtained in this research is described as error-prone, though. Not only is this project a useful application for a real-world problem solving pressing real-world issues, it is also a combination of interesting challenges and questions in the domain of image analysis and machine learning. The original research paper cited above raised the issue of satellite images alone not having high-enough resolution to make confident decisions – the suggested solution would hence add more dimensions to the dataset and verify the image analysis conducted by Marine Traffic APIs or other supplementary sensor data available. This extends the rather straightforward task of image analysis to a more complex domain, where objects in images do not only have to be classified (vessel vs. no vessel) but also located and put into a real-world context (e.g. is the identified vessel a registered ship?).
