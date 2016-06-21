################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../src/FFNN.o \
../src/activation.o \
../src/global.o \
../src/main.o \
../src/read_data.o \
../src/train.o 

CPP_SRCS += \
../src/FFNN.cpp \
../src/activation.cpp \
../src/global.cpp \
../src/main.cpp \
../src/read_data.cpp \
../src/train.cpp 

OBJS += \
./src/FFNN.o \
./src/activation.o \
./src/global.o \
./src/main.o \
./src/read_data.o \
./src/train.o 

CPP_DEPS += \
./src/FFNN.d \
./src/activation.d \
./src/global.d \
./src/main.d \
./src/read_data.d \
./src/train.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


