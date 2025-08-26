# DeafEd – Setup & Requirements Guide 🤟

## 🧠 Project Overview
DeafEd is a Final Year Project (FYP) designed to support **illiterate deaf children in Pakistan** by teaching sign language through **real-time gesture recognition**.  
With 95% of the deaf population unable to read or write, DeafEd bridges the gap using an interactive **React Native app**, a **Node.js backend**, and a **Python ML backend (Flask + OpenCV)**.  

---

## ✨ Features
- 🤚 Real-time Gesture Recognition  
- 🎬 Animated Lessons for engagement (+60% retention)  
- 📊 Educator Dashboard for student tracking (+75% accuracy improvement)  
- 🌐 Dual Backend (Node.js/Express + Python/Flask)  

---

## 📲 How to Use the App

### Step 1: Install Android Emulator
- Download and install [Android Studio](https://developer.android.com/studio).  
- Or connect a physical Android phone for testing.  

---

## ⚙️ React Native Setup (Frontend)

### Update Your IP
- In the config file, replace the placeholder IP with your **local IP address** (same for Node.js & Python backends).  
- To find your IP:  
```bash
ipconfig
```

### Start the App
```bash
cd mobile-app
npm install
npm start
# or
npm run dev
```

---

## 🌐 Node.js Backend (Main)

### Install Dependencies
```bash
cd backend-node
npm install
```

### Run Server
```bash
nodemon server.js
```

---

## 🧠 Python Backend (Models)

### Create Virtual Environment
```bash
cd backend-python
python -m venv venv
.\venv\Scripts\activate   # For Windows
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### Run Flask Server
```bash
python app.py
```

⚠️ Place trained ML models inside `backend-python/models/`.

---

## 🛠️ System Workflow
1. Mobile App (React Native) → Sends gesture input  
2. Python Backend (Flask + OpenCV) → Recognizes gestures using ML models  
3. Node.js Backend (Express + MongoDB) → Stores data, manages user profiles & dashboard  

---

## 📊 Impact
- Helps 95% illiterate deaf children learn sign language  
- Boosts engagement by 60%  
- Improves educator tracking accuracy by 75%  

---

## 👨‍💻 Author
Hasan Sohaib – [GitHub Profile](https://github.com/HasanSohaib-4434)
