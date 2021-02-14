class Review:

  """A review class, contains all bunch of relevant fields"""
  def __init__(self, RECOMMENDATION, COMMENTS, SIGNIFICANCE_SCORES,REVIEWER_CONFIDENCE=None,TITLE=None,):
    self.RECOMMENDATION = RECOMMENDATION
    self.COMMENTS = COMMENTS
    self.CONFIDENCE = REVIEWER_CONFIDENCE
    self.TITLE = TITLE
    self.SIGNIFICANCE_SCORES = SIGNIFICANCE_SCORES


  @staticmethod
  def get_json_string(json_object, string, default=None):
    if string in json_object:
      return json_object[string]
    else:
      return default

    return None

  @staticmethod
  def from_json_object(json_object):
    assert "comments" in json_object
    comments = json_object["comments"]

    recommendation = Review.get_json_string(json_object, "RECOMMENDATION")
    reviewer_confidence = Review.get_json_string(json_object, "CONFIDENCE")
    title = Review.get_json_string(json_object, "TITLE")
    significance_scores = Review.get_json_string(json_object, "SCORES", [])

    return Review(recommendation, comments, significance_scores, reviewer_confidence,title)

    

  def to_json_object(self):
    data = dict()

    data["comments"] = self.get_comments().decode('cp1252', errors='ignore').encode('utf-8')

    if self.RECOMMENDATION is not None:
      data["RECOMMENDATION"] = self.get_recommendation()
    if self.REVIEWER_CONFIDENCE is not None:
      data["CONFIDENCE"] = self.get_reviewer_confidence()
    if self.TITLE is not None:
      data["TITLE"] = self.get_title()
    if self.DATE is not None:
      data["DATE"] = self.get_date()

    return data

  def get_recommendation(self):
    return self.RECOMMENDATION

  def get_comments(self):
    return self.COMMENTS

  def get_reviewer_confidence(self):
    return self.CONFIDENCE

  def get_title(self):
    return self.TITLE


# def __name__ == "__main__":
#   pass
