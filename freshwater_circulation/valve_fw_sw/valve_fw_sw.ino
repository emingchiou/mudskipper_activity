#include <Time.h>
#include <DS1307RTC.h>
#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

int F_I_v_1 = 2; 
int F_I_v_2 = 3; 
int F_O_v_1 = 4; 
int F_O_v_2 = 5; 
int S_O_v_1 = 6; 
int S_O_v_2 = 7; 
int S_I_p = 8;   
#define trigPin A0
#define echoPin A1
int SDcard = 10;
 
File myFile;
   
const unsigned long sec = 1000;
const unsigned long mins = 60 * sec;
const unsigned long hrs = 3600 * sec;

long tidal_period = 6.5*hrs;
long wait = 5*hrs;

void setup(){
  Serial.begin(9600);
  while (!Serial) ; // wait for serial
  delay(200);
  pinMode(SDcard, OUTPUT);
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
    Serial.println(F("RTC success."));
    } 
  else{
    Serial.println(F("RTC failed"));
  return;
  }

  hold();
}    

void loop(){
  check_height();
  
  long t = RTC.get()-28800;
  long a = t+30; //a>t+wait
  Serial.println(RTC.get()-28800);// for 校時
  while(RTC.get()){  
    delay(wait);
    while(RTC.get()){
      long t = RTC.get()-28800;
      if(t >= a){       
        break;
      }  
    }
    fwout(120);
    swin(150);
                             
    a += tidal_period;
    delay(wait);
    while(RTC.get()){
      long t = RTC.get()-28800;
      if(t >= a){
        break;
      }    
    }
    swout(120);
    fwin(120);
   
    a += tidal_period;
  }
}

void w_event(char* event){
  Serial.println(event);
  File myFile = SD.open(F("tidal.txt"), FILE_WRITE);
  tmElements_t tm;
  if (myFile){
    myFile.print(RTC.get()-28800);
    myFile.print(','); 
    myFile.println(event);
    myFile.close();   
  }
  else {
    Serial.println(F("error opening tidal.txt"));
  }  
}

void hold(void){
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
}

void fwout(int delay_second){       
//  fresh water out
  Serial.println("fwout");
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,HIGH);
  digitalWrite(F_O_v_2,LOW);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,LOW);
//  w_event("Sea water in");
  delay(delay_second*sec);    
  hold();
}

void swin(int delay_second){       
//  Sea water in
  Serial.println("swin");
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,HIGH);
//  w_event("Sea water in");
  delay(delay_second*sec);  
  hold(); 
}

void swout(int delay_second){
//  Sea water out
  Serial.println("swout");
  digitalWrite(F_I_v_1,LOW);
  digitalWrite(F_I_v_2,HIGH);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,HIGH);
  digitalWrite(S_O_v_2,LOW);
  digitalWrite(S_I_p,LOW);  
//  w_event("Sea water out");
  delay(delay_second*sec); 
  hold();
}

void fwin(int delay_second){       
//  fresh water in
  Serial.println("fwin");
  digitalWrite(F_I_v_1,HIGH);
  digitalWrite(F_I_v_2,LOW);
  digitalWrite(F_O_v_1,LOW);
  digitalWrite(F_O_v_2,HIGH);
  digitalWrite(S_O_v_1,LOW);
  digitalWrite(S_O_v_2,HIGH);
  digitalWrite(S_I_p,LOW);
//  w_event("Sea water in");
  delay(delay_second*sec); 
  hold();
}

void check_height(void){
  while(echoPin > 0){
    long duration, distance;
    digitalWrite(trigPin, LOW);
    delay(2);
    digitalWrite(trigPin, HIGH);
    delay(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    distance = (duration/2)*0.0346;
    int height = 30 - distance;
    Serial.println(height);
    return height;
  }
}
