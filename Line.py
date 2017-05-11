import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():

    #------------------------------------------------------------
    # Static Variables
    max_history = 5

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.currentx = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

    def updateData(self):
        #Append the current_fit to the history
        self.recent_xfitted.insert(0, self.current_fit)
        self.recent_xfitted = self.recent_xfitted[:min(len(self.recent_xfitted), Line.max_history)]
        # average the new line params
        self.best_fit = np.average(self.recent_xfitted, axis=0)

    def calculateCurvature(self, ploty):
        # Fit new polynomials to x,y in world space
        y_eval = np.max(ploty)
        fit_cr = np.polyfit(ploty * Line.ym_per_pix, self.currentx * Line.xm_per_pix, 2)
        # Calculate the new radii of curvature
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * Line.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])
        # Now our radius of curvature is in meters
        print(self.radius_of_curvature, 'm')