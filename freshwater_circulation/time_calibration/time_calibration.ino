#include <Wire.h>
#include <Time.h>
#include <DS1307RTC.h>

int y; // 年
byte m, d, w, h, mi, s; // 月/日/週/時/分/秒
const byte DS1307_I2C_ADDRESS = 0x68; // DS1307 (I2C) 地址
const byte NubberOfFields = 7; // DS1307 (I2C) 資料範圍


void setup() {
    Serial.begin(9600);
    Serial.println("-------------------");

    Wire.begin();
    setTime(21,5,24,1,13,32,9); // 設定時間：20xx 年 x 月 xx 日 星期x xx 點 xx 分 xx 秒
}

void loop() {
    getTime(); // 取得時間
    digitalClockDisplay(); // 顯示時間
    delay(1000);
}

// BCD 轉 DEC
byte bcdTodec(byte val){
    return ((val / 16 * 10) + (val % 16));
}

// DEC 轉 BCD
byte decToBcd(byte val){
    return ((val / 10 * 16) + (val % 10));
}

// 設定時間
void setTime(byte y, byte m, byte d, byte w, byte h, byte mi, byte s){
    Wire.beginTransmission(DS1307_I2C_ADDRESS);
    Wire.write(0);
    Wire.write(decToBcd(s));
    Wire.write(decToBcd(mi));
    Wire.write(decToBcd(h));
    Wire.write(decToBcd(w));
    Wire.write(decToBcd(d));
    Wire.write(decToBcd(m));
    Wire.write(decToBcd(y));
    Wire.endTransmission();
}

// 取得時間
void getTime(){
    Wire.beginTransmission(DS1307_I2C_ADDRESS);
    Wire.write(0);
    Wire.endTransmission();

    Wire.requestFrom(DS1307_I2C_ADDRESS, NubberOfFields);

    s = bcdTodec(Wire.read() & 0x7f);
    mi = bcdTodec(Wire.read());
    h = bcdTodec(Wire.read() & 0x7f);
    w = bcdTodec(Wire.read());
    d = bcdTodec(Wire.read());
    m = bcdTodec(Wire.read());
    y = bcdTodec(Wire.read()) + 2000;
}

// 顯示時間
void digitalClockDisplay(){
    Serial.print(y);
    Serial.print("/");
    Serial.print(m);
    Serial.print("/");
    Serial.print(d);
    Serial.print(" ( ");
    Serial.print(w);
    Serial.print(" ) ");
    Serial.print(h);
    Serial.print(":");
    Serial.print(mi);
    Serial.print(":");
    Serial.println(s);
}
