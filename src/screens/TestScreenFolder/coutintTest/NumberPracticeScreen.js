import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ImageBackground,
} from "react-native";
import { useRoute } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import BASE_URL2 from "../../../../config2";
import BASE_URL from "../../../../config";

const NumberPracticeScreen = () => {
  const route = useRoute();
  const { number, Username } = route.params;

  const [result, setResult] = useState(null);
  const [isTesting, setIsTesting] = useState(false);

  const startTest = async () => {
    setIsTesting(true);
    setResult(null);

    try {
      const response = await fetch(`${BASE_URL2}/test_gesture-math`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ number }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

      await fetch(`${BASE_URL}/save-counting-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: Username,
          number,
          recognized: data.recognized,
          accuracy: data.accuracy,
          status: data.status,
          timestamp: new Date().toISOString(),
        }),
      });
    } catch (error) {
      setResult({ error: "Error connecting to server" });
      console.error("Request Error:", error);
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <ImageBackground
      source={require("../../../../assets/a.webp")}
      style={styles.backgroundImage}
    >
      <LinearGradient
        colors={["rgba(255, 255, 255, 0.6)", "rgba(255, 255, 255, 0.8)"]}
        style={styles.overlay}
      >
        <View style={styles.container}>
          <Text style={styles.title}>Test Gesture for: {number}</Text>
          <TouchableOpacity
            style={styles.button}
            onPress={startTest}
            disabled={isTesting}
          >
            <Text style={styles.buttonText}>
              {isTesting ? "Testing..." : "Start Test"}
            </Text>
          </TouchableOpacity>

          {result && (
            <Text style={styles.result}>
              {result.status}: {result.recognized} ({result.accuracy}%)
            </Text>
          )}
        </View>
      </LinearGradient>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },
  overlay: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#FF7043",
    marginBottom: 20,
    textAlign: "center",
  },
  button: {
    width: 200,
    height: 50,
    justifyContent: "center",
    alignItems: "center",
    marginVertical: 10,
    borderRadius: 25,
    elevation: 6,
    backgroundColor: "#4FC3F7",
  },
  buttonText: {
    fontSize: 20,
    fontWeight: "bold",
    color: "#fff",
  },
  result: {
    fontSize: 18,
    color: "#4FC3F7",
    marginTop: 20,
    fontWeight: "bold",
  },
});

export default NumberPracticeScreen;
