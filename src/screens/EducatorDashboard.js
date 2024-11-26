import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const EducatorDashboard = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Educator Dashboard</Text>
      <Text style={styles.subtitle}>Welcome to your dashboard!</Text>
      <Text style={styles.content}>Here you can manage students, view reports, and create lessons.</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#3b5998',
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 18,
    color: '#666',
    marginBottom: 20,
  },
  content: {
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
});

export default EducatorDashboard;
