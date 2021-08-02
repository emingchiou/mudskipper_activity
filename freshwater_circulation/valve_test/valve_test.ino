#include <Time.h>
#include <DS1307RTC.h>
#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

int F_O_v_1 = 2; 
int F_O_v_2 = 3; 
int F_I_v_1 = 4; 
int F_I_v_2 = 5; 
int S_O_v_1 = 6; 
int S_O_v_2 = 7; 
int S_I_p = 8;   
int FloatSensor_5 = 5; 
int FloatSensor_15 = 4;
#define trigPin A0
#define echoPin A1
int SDcard = 10;
 
File myFile;
   
const unsigned long sec = 1000;
const unsigned long mins = 60 * sec;
const unsigned long hrs = 3600 * sec;

long tidal_period = 300;//unit = sec
long wait = 2*mins;

void setup(){
  Serial.begin(9600);
  while (!Serial) ; // wait for serial
  delay(200);
//  pinMode(SDcard, OUTPUT);
//  pinMode(FloatSensor_5, INPUT_PULLUP); //Arduino Internal Resistor 10K
//  pinMode(FloatSensor_15, INPUT_PULLUP);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode (F_O_v_1, OUTPUT);
  pinMode (F_O_v_2, OUTPUT);
  pinMode (F_I_v_1, OUTPUT);
  pinMode (F_I_v_2, OUTPUT);
  pinMode (S_O_v_1, OUTPUT);
  pinMode (S_O_v_2, OUTPUT);
  pinMode (S_I_p, OUTPUT);
  
  if (SD.begin()){
    Serial.println(F("SD card success."));} 
  else{
    Serial.println(F("SD card failed"));
  return;
  }
  
  if(RTC.get()){
    Serial.println(F("RTC success."));} 
  else{
    Serial.println(F("RTC failed"));
  return;
  }

//  Hold
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,LOW);
}    

void loop(){
  long t = RTC.get()-28800;
  long a = t+30; //a>t+wait
  while(RTC.get()){
    Serial.println(RTC.get()-28800);// for 校時
    delay(wait);
    while(RTC.get()){
      long t = RTC.get()-28800;
      if(t >= a){       
        break;
      }  
    }
    swin();                         
    a += tidal_period;
    delay(wait);
    while(RTC.get()){
      long t = RTC.get()-28800;
      if(t >= a){
        break;
      }    
    }
    swout();
    a += tidal_period;
  }
}

//void w_event(char* event){
//  Serial.println(event);
//  File myFile = SD.open(F("tidal.txt"), FILE_WRITE);
//  tmElements_t tm;
//  if (myFile){
//    myFile.print(RTC.get()-28800);
//    myFile.print(','); 
//    myFile.println(event);
//    myFile.close();   
//  }
//  else {
//    Serial.println(F("error opening tidal.txt"));
//  }  
//}

void swin(void){       
//  Sea water in
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,HIGH);
//  w_event("Sea water in");
  delay(30*sec);    
//  Hold
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,LOW);
  Serial.println(F("Hold"));  
  delay(10*sec);

// Check
  while(echoPin > 0){
    long duration, distance;
    digitalWrite(trigPin, LOW);
    delay(2);
    digitalWrite(trigPin, HIGH);
    delay(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    distance = (duration/2)*0.0346;
    Serial.println(distance);
   
    if (distance > 19){
      digitalWrite(F_I_v_1,LOW);
      digitalWrite(F_I_v_2,HIGH);
      digitalWrite(F_O_v_1,LOW);
      digitalWrite(F_O_v_2,HIGH);
      digitalWrite(S_O_v_1,LOW);
      digitalWrite(S_O_v_2,HIGH);
      digitalWrite(S_I_p,HIGH);
      delay(3*sec);
      digitalWrite(F_I_v_1,LOW);
      digitalWrite(F_I_v_2,HIGH);
      digitalWrite(F_O_v_1,LOW);
      digitalWrite(F_O_v_2,HIGH);
      digitalWrite(S_O_v_1,LOW);
      digitalWrite(S_O_v_2,HIGH);
      digitalWrite(S_I_p,LOW);
      Serial.println(F("add water"));
    }
    else if (distance < 19){
      digitalWrite(F_I_v_1,LOW);
      digitalWrite(F_I_v_2,HIGH);
      digitalWrite(F_O_v_1,LOW);
      digitalWrite(F_O_v_2,HIGH);
      digitalWrite(S_O_v_1,HIGH);
      digitalWrite(S_O_v_2,LOW);
      digitalWrite(S_I_p,LOW); 
      delay(5*sec);
      digitalWrite(F_I_v_1,LOW);
      digitalWrite(F_I_v_2,HIGH);
      digitalWrite(F_O_v_1,LOW);
      digitalWrite(F_O_v_2,HIGH);
      digitalWrite(S_O_v_1,LOW);
      digitalWrite(S_O_v_2,HIGH);
      digitalWrite(S_I_p,LOW);
      Serial.println(F("out water"));
    }
    else{
      break;
    }
    delay(10*sec);
  }

}

void swout(void){
//  Sea water out
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,HIGH);
  digitalWrite(S_O_v_2,LOW);
  digitalWrite(S_I_p,LOW);  
//  w_event("Sea water out");
  delay(1*mins);  
//  Hold
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,LOW);
  Serial.println(F("Hold"));
}
