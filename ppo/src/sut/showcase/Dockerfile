FROM node:18-alpine

WORKDIR /app

COPY src/sut/showcase/ ./

RUN npm install

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
