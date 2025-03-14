const express = require("express");
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const cors = require("cors");
const studentController = require("./controllers/studentController");
const EducatorController = require("./controllers/educatorController");
const sectionController = require("./controllers/sectionController");
const educatorStudentController = require("./controllers/educatorStudentController");

require("dotenv").config();

const {
  saveEnglishTestResult,
} = require("./controllers/englishResultController");
const { getEnglishResults } = require("./controllers/englishResultController");

const { saveUrduTestResult } = require("./controllers/urduResultController");
const { getUrduResults } = require("./controllers/urduResultController");

const {
  saveCountingTestResult,
} = require("./controllers/countingResultController");
const {
  getCountingResults,
} = require("./controllers/countingResultController");

const {
  generateOtp,
  verifyOtp,
  signup,
} = require("./controllers/signupController");

const {
  login,
  forgotPassword,
  verifyResetOtp,
  resetPassword,
} = require("./controllers/loginController");

const {
  getAlphabetsProgress,
  updateAlphabetsProgress,
} = require("./controllers/AlphabetProgressController");

const {
  getFoodProgress,
  updateFoodProgress,
} = require("./controllers/FoodProgressController");
const {
  getRelationProgress,
  updateRelationProgress,
} = require("./controllers/RelationProgressController");

const {
  getUrduAlphabetsProgress,
  updateUrduAlphabetsProgress,
} = require("./controllers/UrduAlphabetsProgressController");
const app = express();
app.use(cors());
app.use(bodyParser.json());

const {
  getCountingProgress,
  updateCountingProgress,
} = require("./controllers/CountingProgressController");

// MongoDB Connection
mongoose
  .connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("Connected to MongoDB"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Signup Routes
app.post("/generate-otp", generateOtp);
app.post("/verify-otp", verifyOtp);
app.post("/signup", signup);

// Login Routes
app.post("/login", login);
app.post("/forgot-password", forgotPassword);
app.post("/verify-reset-otp", verifyResetOtp);
app.post("/reset-password", resetPassword);

//retriving student data
app.get("/students/:email", studentController.retrieveStudentData);

//retriving educator data
app.get("/educators/:email", EducatorController.retrieveEducatortData);

//delete profiles
app.delete("/delete-educator/:educatorName", EducatorController.deleteEducator);

//delete profiles
app.delete("/delete-student/:studentName", studentController.deleteStudent);

// Alphabet Progress Routes
app.get("/alphabetsProgress", getAlphabetsProgress);
app.post("/alphabetsProgress", updateAlphabetsProgress);

app.get("/urduAlphabetsProgress", getUrduAlphabetsProgress);
app.post("/urduAlphabetsProgress", updateUrduAlphabetsProgress);

app.get("/countingProgress", getCountingProgress);
app.post("/countingProgress", updateCountingProgress);

app.get("/foodProgress", getFoodProgress);
app.post("/foodProgress", updateFoodProgress);

app.get("/relationProgress", getRelationProgress);
app.post("/relationProgress", updateRelationProgress);

//updating profiles
app.put("/students", studentController.updateStudentProfile);
app.put("/educators", EducatorController.updateEducatorFullName);

//section
app.get("/sections", sectionController.getSections);
app.post("/sections", sectionController.createSection);
app.delete("/sections/:id", sectionController.deleteSection);

//educatorstudents
app.get("/educatorstudents", educatorStudentController.getStudentsBySection);
app.post("/educatorstudents", educatorStudentController.addStudent);
app.delete("/educatorstudents/:id", educatorStudentController.deleteStudent);

//results
app.post("/save_english_test_result", saveEnglishTestResult);
app.get("/get_english_results", getEnglishResults);

app.post("/save-urdu-test", saveUrduTestResult);
app.get("/get_urdu_results", getUrduResults);

app.post("/save-counting-test", saveCountingTestResult);
app.get("/get_counting_results", getCountingResults);

// Start Server
app.listen(process.env.PORT, () => {
  console.log(`Server is running on port ${process.env.PORT}`);
});
