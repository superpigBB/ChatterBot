CREATE DATABASE  IF NOT EXISTS `test` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `test`;
-- MySQL dump 10.13  Distrib 8.0.17, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: test
-- ------------------------------------------------------
-- Server version	8.0.17

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `training_table`
--

DROP TABLE IF EXISTS `training_table`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `training_table` (
  `training_id` varchar(100) NOT NULL,
  `tag` longtext,
  `pattern` longtext,
  `response` longtext,
  PRIMARY KEY (`training_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `training_table`
--

LOCK TABLES `training_table` WRITE;
/*!40000 ALTER TABLE `training_table` DISABLE KEYS */;
INSERT INTO `training_table` VALUES ('1','greeting','hi','hello'),('10','query_display','autocomplete data','SELECT * FROM test.output;'),('11','query_download','download autocomplete data','SELECT * FROM test.output;'),('12','query_download','autocomplete data','SELECT * FROM test.output;'),('13','url','about autocomplete','https://en.wikipedia.org/wiki/Autocomplete'),('14','url','tell me more autocomplete','https://en.wikipedia.org/wiki/Autocomplete'),('15','url','about DUDL','https://maportalprd01.corp.emc.com/MRES_Metrics_And_Analytics/ViewViz.aspx?id=1241'),('16','url','DUDL report','https://maportalprd01.corp.emc.com/MRES_Metrics_And_Analytics/ViewViz.aspx?id=1241'),('17','url','i want DUDL report','https://maportalprd01.corp.emc.com/MRES_Metrics_And_Analytics/ViewViz.aspx?id=1241'),('2','greeting','hello','hello'),('3','goodbye','goodbye','talk to you later'),('4','goodbye','cya','talk to you later'),('5','age','how old are you','im 18 years old'),('6','age','what is your age','im 18 years old'),('7','name','what is your name','You can call me Pylot'),('8','name','whats your name','You can call me Pylot'),('9','query_display','display autocomplete data','SELECT * FROM test.output;');
/*!40000 ALTER TABLE `training_table` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2019-10-21 15:29:21
