import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ImageBackground,
} from "react-native";
import axios from "axios";
import BASE_URL from "../../../config";

const UrduLessonScreen = ({ navigation, route }) => {
  const urduAlphabet = [
    "ا",
    "ب",
    "پ",
    "ت",
    "ٹ",
    "ث",
    "ج",
    "چ",
    "ح",
    "خ",
    "د",
    "ڈ",
    "ذ",
    "ر",
    "ڑ",
    "ز",
    "ژ",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ک",
    "گ",
    "ل",
    "م",
    "ن",
    "ں",
    "و",
    "ہ",
    "ء",
    "ی",
    "ے",
  ];
  const { Username } = route.params;
  const [openedLetters, setOpenedLetters] = useState([]);

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const response = await axios.get(
          `${BASE_URL}/urduAlphabetsProgress?username=${Username}`
        );
        setOpenedLetters(response.data.openedLetters || []);
      } catch (error) {
        console.error("Error fetching progress:", error);
      }
    };

    fetchProgress();
  }, []);

  const handleLetterClick = async (letter) => {
    try {
      await axios.post(`${BASE_URL}/urduAlphabetsProgress`, {
        username: Username,
        letter,
      });
      setOpenedLetters((prev) => [...prev, letter]);
      navigation.navigate("UrduAnimations", { letter });
    } catch (error) {
      console.error("Error updating progress:", error);
    }
  };

  return (
    <ImageBackground
      source={require("../../../assets/a.webp")}
      style={styles.backgroundImage}
    >
      <View style={styles.container}>
        <Text style={styles.title}>اردو اسباق</Text>
        <Text style={styles.subtitle}>سیکھنے کے لیے منتخب کریں</Text>

        <ScrollView contentContainerStyle={styles.buttonsContainer}>
          {urduAlphabet.map((letter, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.button,
                openedLetters.includes(letter)
                  ? styles.buttonOpened
                  : styles.buttonDefault,
              ]}
              onPress={() => handleLetterClick(letter)}
            >
              <Text style={styles.buttonText}>{letter}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "rgba(255, 255, 255, 0.6)", // Light overlay for readability
    justifyContent: "flex-start",
    alignItems: "center",
    padding: 20,
  },
  backgroundImage: {
    flex: 1,
    resizeMode: "cover",
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#FF7043",
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 18,
    color: "#000",
    marginBottom: 20,
  },
  buttonsContainer: {
    alignItems: "center",
    paddingBottom: 20,
  },
  button: {
    width: 200,
    height: 50,
    justifyContent: "center",
    alignItems: "center",
    marginVertical: 10,
    borderRadius: 25,
    elevation: 6,
  },
  buttonDefault: {
    backgroundColor: "#4FC3F7",
  },
  buttonOpened: {
    backgroundColor: "#BDBDBD",
  },
  buttonText: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#fff",
  },
});

export default UrduLessonScreen;
