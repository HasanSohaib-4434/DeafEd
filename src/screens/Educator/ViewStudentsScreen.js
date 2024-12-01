import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import DropDownPicker from 'react-native-dropdown-picker';

const ViewStudentsScreen = ({ route }) => {
  const { educatorUsername } = route.params;
  const [sections, setSections] = useState([]);
  const [students, setStudents] = useState([]);
  const [selectedSection, setSelectedSection] = useState(null);
  const [loadingSections, setLoadingSections] = useState(false);
  const [loadingStudents, setLoadingStudents] = useState(false);

  const [sectionDropdownOpen, setSectionDropdownOpen] = useState(false);

  useEffect(() => {
    fetchSections();
  }, []);

  const fetchSections = async () => {
    setLoadingSections(true);
    try {
      const response = await fetch(
        `http://192.168.1.117:5000/sections?educatorUsername=${educatorUsername}`
      );
      const data = await response.json();
      if (response.ok) {
        setSections(data.sections.map((section) => ({ label: section, value: section })));
      } else {
        Alert.alert('Error', 'Unable to fetch sections');
      }
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'An error occurred while fetching sections');
    } finally {
      setLoadingSections(false);
    }
  };

  const fetchStudents = async () => {
    if (!selectedSection) {
      Alert.alert('Validation Error', 'Please select a section first.');
      return;
    }
    setLoadingStudents(true);
    try {
      const response = await fetch(
        `http://192.168.1.117:5000/students?educatorUsername=${educatorUsername}&section=${selectedSection}`
      );
      const data = await response.json();
      if (response.ok) {
        setStudents(data.students.map((student) => ({ label: student, value: student })));
      } else {
        Alert.alert('Error', 'Unable to fetch students');
      }
    } catch (error) {
      console.error(error);
      Alert.alert('Error', 'An error occurred while fetching students');
    } finally {
      setLoadingStudents(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>View Students</Text>

      <DropDownPicker
        items={sections}
        open={sectionDropdownOpen}
        setOpen={setSectionDropdownOpen}
        value={selectedSection}
        setValue={setSelectedSection}
        placeholder="Select a section"
        loading={loadingSections}
        style={styles.dropdown}
      />
      {!sectionDropdownOpen && (
        <TouchableOpacity style={styles.findButton} onPress={fetchStudents}>
          <Text style={styles.buttonText}>View Students</Text>
        </TouchableOpacity>
      )}

      {students.length > 0 && (
        <View style={styles.studentList}>
          <Text style={styles.listTitle}>Students in Section {selectedSection}:</Text>
          {students.map((student, index) => (
            <Text key={index} style={styles.studentItem}>
              {student.label}
            </Text>
          ))}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 20,
  },
  dropdown: {
    marginBottom: 15,
    zIndex: 10, 
  },
  findButton: {
    backgroundColor: '#4CAF50',
    padding: 10,
    borderRadius: 5,
    marginBottom: 20,
  },
  studentList: {
    marginTop: 20,
  },
  listTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  studentItem: {
    fontSize: 16,
    marginVertical: 2,
  },
  buttonText: {
    color: '#fff',
    textAlign: 'center',
  },
});

export default ViewStudentsScreen;
