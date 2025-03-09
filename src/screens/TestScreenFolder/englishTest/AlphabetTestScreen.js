import React from "react";
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  ImageBackground,
} from "react-native";
import { useNavigation } from "@react-navigation/native";

const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const AlphabetTestScreen = ({ route }) => {
  const navigation = useNavigation();
  const { Username } = route.params;

  return (
    <ImageBackground
      source={require("../../../../assets/a.webp")} // Background image added
      style={styles.backgroundImage}
    >
      <View style={styles.container}>
        <Text style={styles.title}>Select an Alphabet</Text>

        <ScrollView contentContainerStyle={styles.buttonsContainer}>
          {alphabet.map((letter, index) => (
            <TouchableOpacity
              key={index}
              style={styles.button}
              onPress={() =>
                navigation.navigate("LetterPracticeScreen", {
                  letter,
                  Username,
                })
              }
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
  backgroundImage: {
    flex: 1,
    width: "100%",
    height: "100%",
    resizeMode: "cover",
  },
  container: {
    flex: 1,
    backgroundColor: "rgba(255, 255, 255, 0.6)", // Light overlay for readability
    justifyContent: "flex-start",
    alignItems: "center",
    padding: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    color: "#FF7043",
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
    backgroundColor: "#4FC3F7",
  },
  buttonText: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#fff",
  },
});

export default AlphabetTestScreen;
