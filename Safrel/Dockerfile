FROM openjdk:17-jdk-slim

WORKDIR /app

COPY target/my-app.jar my-app.jar
COPY src/main/java/performancetest/util/StateDetection.fcl.rtf .

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "echo $HOSTNAME > /app/my-app && java -jar my-app.jar"]

CMD ["java", "-jar", "my-app.jar"]
