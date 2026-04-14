#include <SparkFun_AS7343.h>
#include <Arduino.h>
#include <Wire.h>
#include "model.h" // Includes the exported Random Forest model

SfeAS7343ArdI2C mySensor;
uint16_t myData[ksfAS7343NumChannels];

#define RXD2 16
#define TXD2 17
#define NUM_SPECTRAL 13

// ── Helper: Standard Deviation ──────────────────────
float calculate_std(float* arr, int len, float mean) {
    float sum = 0.0;
    for (int i = 0; i < len; i++) sum += pow(arr[i] - mean, 2);
    return sqrt(sum / len);
}

// ── pH Reader ───────────────────────────────────────
float read_ph_live() {
    unsigned long start = millis();
    while (millis() - start < 1000) {
        if (Serial2.available()) {
            String line = Serial2.readStringUntil('\n');
            line.toLowerCase();
            int idx = line.indexOf("ph:");
            if (idx != -1) {
                int numStart = idx + 3;
                int sep = line.indexOf("||", numStart);
                String phStr = (sep != -1) ? line.substring(numStart, sep) : line.substring(numStart);
                phStr.trim();
                float val = phStr.toFloat();
                if (val > 0.0) return val;
            }
        }
    }
    return -1.0;
}

float read_ph_averaged(int n = 7) {
    float sum = 0.0; int count = 0;
    while (count < n) {
        float ph = read_ph_live();
        if (ph > 0.0) { sum += ph; count++; }
        else Serial.println("  [pH] Bad reading, retrying...");
    }
    return sum / n;
}

// ── Read Spectral into ordered float array ──────────
bool read_spectral(float* ch) {
    mySensor.ledOn();
    mySensor.setAgain(AGAIN_1);
    delay(300);

    if (mySensor.readSpectraDataFromSensor() == false) {
        Serial.println("ERR:SPECTRAL_FAIL");
        mySensor.ledOff();
        return false;
    }

    ch[0]  = mySensor.getChannelData(CH_PURPLE_F1_405NM);
    ch[1]  = mySensor.getChannelData(CH_DARK_BLUE_F2_425NM);
    ch[2]  = mySensor.getChannelData(CH_BLUE_FZ_450NM);
    ch[3]  = mySensor.getChannelData(CH_LIGHT_BLUE_F3_475NM);
    ch[4]  = mySensor.getChannelData(CH_BLUE_F4_515NM);
    ch[5]  = mySensor.getChannelData(CH_GREEN_F5_550NM);
    ch[6]  = mySensor.getChannelData(CH_GREEN_FY_555NM);
    ch[7]  = mySensor.getChannelData(CH_ORANGE_FXL_600NM);
    ch[8]  = mySensor.getChannelData(CH_BROWN_F6_640NM);
    ch[9]  = mySensor.getChannelData(CH_RED_F7_690NM);
    ch[10] = mySensor.getChannelData(CH_DARK_RED_F8_745NM);
    ch[11] = mySensor.getChannelData(CH_VIS_1);
    ch[12] = mySensor.getChannelData(CH_NIR_855NM);

    mySensor.ledOff();
    return true;
}

// ── Main Dataset Collection Function ────────────────
void take_and_send_reading() {
    float ch[NUM_SPECTRAL] = {0};
    if (!read_spectral(ch)) {
        Serial.println("ERR:READ_FAILED");
        return;
    }

    float avg_ph = read_ph_averaged(7);
    if (avg_ph < 0) {
        Serial.println("ERR:PH_FAILED");
        return;
    }

    float total = 0, max_val = 0, min_val = 1e6;
    for (int i = 0; i < NUM_SPECTRAL; i++) {
        total += ch[i];
        if (ch[i] > max_val) max_val = ch[i];
        if (ch[i] < min_val) min_val = ch[i];
    }
    if (total < 1.0) total = 1.0;

    float mean_ch        = total / NUM_SPECTRAL;
    float spectral_width = calculate_std(ch, NUM_SPECTRAL, mean_ch);
    float log_intensity  = log(total + 1e-8);
    float peak_to_mean   = max_val / (mean_ch + 1e-8);
    float spec_contrast  = (max_val - min_val) / (max_val + 1e-8);
    float ph_dev         = fabs(avg_ph - 6.68);

    float ratio_FY_F1    = ch[6]  / (ch[0]  + 1e-8);
    float ratio_F7_FY    = ch[9]  / (ch[6]  + 1e-8);
    float ratio_F4_NIR   = ch[4]  / (ch[12] + 1e-8);
    float blue_nir_ratio = (ch[0] + ch[1]) / (ch[9] + ch[12] + 1e-8);

    float entropy = 0;
    for (int i = 0; i < NUM_SPECTRAL; i++) {
        float n = ch[i] / total;
        entropy -= n * log(n + 1e-8);
    }

    float int_per_width  = total / (spectral_width + 1e-8);
    float int_per_peak   = total / (peak_to_mean + 1e-8);
    float ph_x_intensity = avg_ph * log_intensity;
    float ph_x_entropy   = avg_ph * entropy;

    // Map features into model input array exactly according to model.h indices
    float x[N_FEATURES];
    int f_idx = 0;

    // 0-12: Raw channels
    for (int i = 0; i < NUM_SPECTRAL; i++) x[f_idx++] = ch[i];
    
    // 13-18: pH and basic derived
    x[f_idx++] = avg_ph;
    x[f_idx++] = total;
    x[f_idx++] = ph_dev;
    x[f_idx++] = ratio_FY_F1;
    x[f_idx++] = ratio_F7_FY;
    x[f_idx++] = ratio_F4_NIR;

    // 19-31: Normalized channels
    for (int i = 0; i < NUM_SPECTRAL; i++) x[f_idx++] = ch[i] / total;

    // 32-41: Spectral features
    x[f_idx++] = log_intensity;
    x[f_idx++] = spectral_width;
    x[f_idx++] = peak_to_mean;
    x[f_idx++] = spec_contrast;
    x[f_idx++] = entropy;
    x[f_idx++] = blue_nir_ratio;
    x[f_idx++] = int_per_width;
    x[f_idx++] = int_per_peak;
    x[f_idx++] = ph_x_intensity;
    x[f_idx++] = ph_x_entropy;

    // Security check to avoid buffer/model misalignment
    if (f_idx != N_FEATURES) {
        Serial.println("ERR:FEATURE_MAP_MISMATCH");
        return;
    }

    // Inference
    int prediction_class = predict_adulteration(x);
    const char* prediction_label = class_name(prediction_class);

    // Output Prediction
    Serial.print("PREDICTION: ");
    Serial.print(prediction_class);
    Serial.print(" -> ");
    Serial.println(prediction_label);

    // Output Features Vector (Debugging/Logging)
    Serial.print("DATA:");
    for (int i = 0; i < N_FEATURES; i++) {
        Serial.print(x[i], 6);
        if (i < N_FEATURES - 1) Serial.print(",");
    }
    Serial.println();
}

// ── Setup ────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("=== AS7343 Inference ===");

    Wire.begin();

    if (mySensor.begin() == false) {
        while (1) { Serial.println("ERR:AS7343 NOT FOUND - Check QWIIC/I2C"); delay(2000); }
    }
    if (mySensor.powerOn() == false) {
        while (1) { Serial.println("ERR:POWER ON FAILED"); delay(2000); }
    }
    if (mySensor.setAutoSmux(AUTOSMUX_18_CHANNELS) == false) {
        while (1) { Serial.println("ERR:AUTOSMUX FAILED"); delay(2000); }
    }
    if (mySensor.enableSpectralMeasurement() == false) {
        while (1) { Serial.println("ERR:SPECTRAL ENABLE FAILED"); delay(2000); }
    }

    Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);

    Serial.println("READY");
    Serial.println("Send 'S' to trigger a reading.");
}

// ── Loop ─────────────────────────────────────────────
void loop() {
    if (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == 'S' || cmd == 's') {
            Serial.println("INFO:Taking reading...");
            take_and_send_reading();
        }
    }
}